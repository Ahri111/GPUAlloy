"""
MCMC Optimization for Perovskite Structures with PyTorch

스왑 → 에너지 평가 → MCMC → 반복 파이프라인

핵심: 스왑 시 atom_types만 변경, edge_index는 유지됨 (결정 구조 불변)

Usage:
    # Single temperature MCMC
    python mcmc_pytorch.py --ckpt ./checkpoints/best.pth --n_structures 10000 --temperature 0.1

    # Parallel tempering (multiple temperatures)
    python mcmc_pytorch.py --mode parallel_tempering --ckpt ./checkpoints/best.pth \
        --n_structures 100000 --n_temps 100 --temp_min 0.01 --temp_max 1.0

    # Benchmark MCMC speed
    python mcmc_pytorch.py --mode benchmark
"""

import argparse
import time
from typing import Optional, Dict, Tuple, List
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from swap_utils import sample_sublattice_swap, swap_by_idx

print("=" * 60)
print(f"PyTorch MCMC Pipeline")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name()}")
print("=" * 60)


# =============================================================================
# Constants
# =============================================================================

ATOM_TYPES = {"Sr": 0, "Ti": 1, "Fe": 2, "O": 3, "VO": 4}
ATOMIC_NUMBERS = {"Sr": 38, "Ti": 22, "Fe": 26, "O": 8, "VO": 0}


# =============================================================================
# Structure Generation
# =============================================================================

def generate_structures(
    batch_size: int,
    n_Sr: int,
    n_Ti: int,
    n_Fe: int,
    n_O: int,
    n_VO: int,
    device: str = 'cuda',
) -> torch.Tensor:
    """Generate random perovskite structures.

    Args:
        batch_size: Number of structures to generate
        n_Sr, n_Ti, n_Fe, n_O, n_VO: Number of each atom type

    Returns:
        atom_types: (batch_size, N) tensor of atom type indices
    """
    N = n_Sr + n_Ti + n_Fe + n_O + n_VO

    # Create base template
    template = torch.cat([
        torch.full((n_Sr,), ATOM_TYPES["Sr"], dtype=torch.long),
        torch.full((n_Ti,), ATOM_TYPES["Ti"], dtype=torch.long),
        torch.full((n_Fe,), ATOM_TYPES["Fe"], dtype=torch.long),
        torch.full((n_O,), ATOM_TYPES["O"], dtype=torch.long),
        torch.full((n_VO,), ATOM_TYPES["VO"], dtype=torch.long),
    ]).to(device)

    # Expand to batch
    structures = template.unsqueeze(0).expand(batch_size, -1).clone()

    # Shuffle B-site (Ti, Fe) for each structure
    b_start = n_Sr
    b_end = n_Sr + n_Ti + n_Fe
    for i in range(batch_size):
        perm = torch.randperm(n_Ti + n_Fe, device=device) + b_start
        structures[i, b_start:b_end] = structures[i, perm]

    # Shuffle O-site (O, VO) for each structure
    o_start = b_end
    o_end = N
    for i in range(batch_size):
        perm = torch.randperm(n_O + n_VO, device=device) + o_start
        structures[i, o_start:o_end] = structures[i, perm]

    return structures


@torch.no_grad()
def generate_structures_fast(
    batch_size: int,
    n_Sr: int,
    n_Ti: int,
    n_Fe: int,
    n_O: int,
    n_VO: int,
    device: str = 'cuda',
) -> torch.Tensor:
    """Faster structure generation using argsort trick."""
    N = n_Sr + n_Ti + n_Fe + n_O + n_VO
    n_B = n_Ti + n_Fe
    n_O_total = n_O + n_VO

    # Create base template
    template = torch.cat([
        torch.full((n_Sr,), ATOM_TYPES["Sr"], dtype=torch.long, device=device),
        torch.full((n_Ti,), ATOM_TYPES["Ti"], dtype=torch.long, device=device),
        torch.full((n_Fe,), ATOM_TYPES["Fe"], dtype=torch.long, device=device),
        torch.full((n_O,), ATOM_TYPES["O"], dtype=torch.long, device=device),
        torch.full((n_VO,), ATOM_TYPES["VO"], dtype=torch.long, device=device),
    ])

    structures = template.unsqueeze(0).expand(batch_size, -1).clone()

    # Shuffle B-sites using argsort
    b_start, b_end = n_Sr, n_Sr + n_B
    rand_b = torch.rand(batch_size, n_B, device=device)
    perm_b = rand_b.argsort(dim=1) + b_start
    b_section = structures[:, b_start:b_end].clone()
    structures[:, b_start:b_end] = b_section.gather(1, perm_b - b_start)

    # Shuffle O-sites using argsort
    o_start, o_end = b_end, N
    rand_o = torch.rand(batch_size, n_O_total, device=device)
    perm_o = rand_o.argsort(dim=1) + o_start
    o_section = structures[:, o_start:o_end].clone()
    structures[:, o_start:o_end] = o_section.gather(1, perm_o - o_start)

    return structures


# =============================================================================
# Atom Feature Lookup
# =============================================================================

def load_atom_features(path: str = 'atom_init.json') -> torch.Tensor:
    """Load atom feature lookup table."""
    with open(path, 'r') as f:
        atom_features = json.load(f)

    max_z = max(int(k) for k in atom_features.keys())
    fea_dim = len(next(iter(atom_features.values())))

    table = torch.zeros(max_z + 2, fea_dim)
    for z_str, fea in atom_features.items():
        table[int(z_str)] = torch.tensor(fea)

    return table


def atom_types_to_features(
    atom_types: torch.Tensor,
    feature_table: torch.Tensor,
    type_to_z: torch.Tensor,
) -> torch.Tensor:
    """Convert atom type indices to features.

    Args:
        atom_types: (batch_size, N) atom type indices
        feature_table: (max_z+1, fea_dim) feature lookup
        type_to_z: (5,) type index to atomic number

    Returns:
        atom_fea: (batch_size, N, fea_dim)
    """
    atomic_nums = type_to_z[atom_types]  # (batch_size, N)
    atom_fea = feature_table[atomic_nums]  # (batch_size, N, fea_dim)
    return atom_fea


# =============================================================================
# MCMC Step
# =============================================================================

@torch.no_grad()
def mcmc_step(
    atom_types: torch.Tensor,       # (batch_size, N)
    current_energy: torch.Tensor,   # (batch_size,)
    temperature: torch.Tensor,      # (batch_size,) or scalar
    b_site_mask: torch.Tensor,      # (N,)
    o_site_mask: torch.Tensor,      # (N,)
    energy_fn,                       # Function: atom_types -> energies
    scores: Optional[torch.Tensor] = None,  # (batch_size, N) for guided swap
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Single MCMC step with Metropolis-Hastings acceptance.

    Args:
        atom_types: Current atom configurations
        current_energy: Current energies
        temperature: Temperature(s) for acceptance
        b_site_mask: Boolean mask for B-site atoms
        o_site_mask: Boolean mask for O-site atoms
        energy_fn: Function to evaluate energy from atom_types
        scores: Optional scores for guided swap

    Returns:
        new_atom_types: Updated configurations
        new_energy: Updated energies
        accepted: Boolean mask of accepted moves
        accept_prob: Acceptance probabilities
    """
    batch_size = atom_types.shape[0]
    device = atom_types.device

    # 1. Propose swap (randomly choose B-site or O-site)
    do_b_site = torch.rand(1, device=device).item() < 0.5

    if do_b_site:
        proposed, _ = sample_sublattice_swap(
            atom_types, b_site_mask,
            ATOM_TYPES["Ti"], ATOM_TYPES["Fe"],
            scores=scores
        )
    else:
        proposed, _ = sample_sublattice_swap(
            atom_types, o_site_mask,
            ATOM_TYPES["O"], ATOM_TYPES["VO"],
            scores=scores
        )

    # 2. Evaluate proposed energy
    proposed_energy = energy_fn(proposed)

    # 3. Metropolis-Hastings acceptance
    delta_E = proposed_energy - current_energy

    # Accept probability: min(1, exp(-dE/T))
    accept_prob = torch.clamp(torch.exp(-delta_E / temperature), max=1.0)

    # Random acceptance
    u = torch.rand(batch_size, device=device)
    accepted = u < accept_prob

    # 4. Update based on acceptance
    new_atom_types = torch.where(
        accepted.unsqueeze(-1),
        proposed,
        atom_types
    )
    new_energy = torch.where(accepted, proposed_energy, current_energy)

    return new_atom_types, new_energy, accepted, accept_prob


# =============================================================================
# MCMC Loop
# =============================================================================

@torch.no_grad()
def run_mcmc(
    init_atom_types: torch.Tensor,
    energy_fn,
    b_site_mask: torch.Tensor,
    o_site_mask: torch.Tensor,
    temperature: float,
    n_steps: int,
    verbose: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, List[float], List[float]]:
    """Run MCMC optimization.

    Args:
        init_atom_types: (batch_size, N) initial configurations
        energy_fn: Function to evaluate energy
        b_site_mask, o_site_mask: Sublattice masks
        temperature: MCMC temperature
        n_steps: Number of MCMC steps

    Returns:
        final_atom_types: Final configurations
        final_energy: Final energies
        energy_history: Mean energy at each step
        accept_history: Acceptance rate at each step
    """
    batch_size = init_atom_types.shape[0]
    device = init_atom_types.device

    # Initialize
    atom_types = init_atom_types.clone()
    energy = energy_fn(atom_types)
    temp_tensor = torch.full((batch_size,), temperature, device=device)

    # Track best
    best_types = atom_types.clone()
    best_energy = energy.clone()

    # History
    energy_history = []
    accept_history = []
    total_accepted = torch.zeros(batch_size, device=device)

    for step in range(n_steps):
        atom_types, energy, accepted, _ = mcmc_step(
            atom_types, energy, temp_tensor,
            b_site_mask, o_site_mask,
            energy_fn
        )

        # Update best
        improved = energy < best_energy
        best_types = torch.where(improved.unsqueeze(-1), atom_types, best_types)
        best_energy = torch.where(improved, energy, best_energy)

        total_accepted += accepted.float()

        if step % 100 == 0:
            mean_energy = best_energy.mean().item()
            accept_rate = total_accepted.mean().item() / (step + 1)
            energy_history.append(mean_energy)
            accept_history.append(accept_rate)

            if verbose:
                print(f"Step {step:5d}: Mean best energy = {mean_energy:.4f}, "
                      f"Accept rate = {accept_rate:.3f}")

    return best_types, best_energy, energy_history, accept_history


# =============================================================================
# Parallel Tempering
# =============================================================================

@torch.no_grad()
def run_parallel_tempering(
    init_atom_types: torch.Tensor,  # (n_temps, structures_per_temp, N)
    energy_fn,
    b_site_mask: torch.Tensor,
    o_site_mask: torch.Tensor,
    temperatures: torch.Tensor,     # (n_temps,)
    n_steps: int,
    exchange_every: int = 10,
    verbose: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, List[float]]:
    """Run parallel tempering MCMC.

    Args:
        init_atom_types: (n_temps, structures_per_temp, N)
        energy_fn: Energy evaluation function
        temperatures: (n_temps,) temperature ladder
        n_steps: Number of MCMC steps
        exchange_every: Attempt replica exchange every N steps

    Returns:
        best_types: Best configurations found
        best_energy: Best energies found
        energy_history: History of best mean energy
    """
    n_temps, structures_per_temp, N = init_atom_types.shape
    device = init_atom_types.device

    # Initialize
    atom_types = init_atom_types.clone()

    # Evaluate initial energies
    flat_types = atom_types.view(-1, N)
    flat_energy = energy_fn(flat_types)
    energy = flat_energy.view(n_temps, structures_per_temp)

    # Temperature tensor for each structure
    temp_per_struct = temperatures.unsqueeze(1).expand(-1, structures_per_temp)

    # Track best (from lowest temperature)
    best_types = atom_types[0].clone()
    best_energy = energy[0].clone()

    energy_history = []

    for step in range(n_steps):
        # 1. MCMC step at each temperature
        flat_types = atom_types.view(-1, N)
        flat_energy = energy.view(-1)
        flat_temp = temp_per_struct.reshape(-1)

        flat_types, flat_energy, accepted, _ = mcmc_step(
            flat_types, flat_energy, flat_temp,
            b_site_mask, o_site_mask,
            energy_fn
        )

        atom_types = flat_types.view(n_temps, structures_per_temp, N)
        energy = flat_energy.view(n_temps, structures_per_temp)

        # 2. Replica exchange (every exchange_every steps)
        if step % exchange_every == 0 and step > 0:
            for i in range(n_temps - 1):
                # Exchange probability between adjacent temperatures
                beta_i = 1.0 / temperatures[i]
                beta_j = 1.0 / temperatures[i + 1]

                for j in range(structures_per_temp):
                    delta = (beta_i - beta_j) * (energy[i, j] - energy[i + 1, j])
                    accept_prob = min(1.0, torch.exp(delta).item())

                    if torch.rand(1).item() < accept_prob:
                        # Swap replicas
                        atom_types[i, j], atom_types[i + 1, j] = \
                            atom_types[i + 1, j].clone(), atom_types[i, j].clone()
                        energy[i, j], energy[i + 1, j] = \
                            energy[i + 1, j].clone(), energy[i, j].clone()

        # Update best (from lowest temperature)
        improved = energy[0] < best_energy
        best_types = torch.where(improved.unsqueeze(-1), atom_types[0], best_types)
        best_energy = torch.where(improved, energy[0], best_energy)

        if step % 100 == 0:
            mean_best = best_energy.mean().item()
            energy_history.append(mean_best)
            if verbose:
                print(f"Step {step:5d}: Best mean energy = {mean_best:.4f}")

    return best_types, best_energy, energy_history


# =============================================================================
# Benchmark
# =============================================================================

def run_benchmark(args):
    """Benchmark MCMC operations."""
    print("\n" + "=" * 60)
    print("PyTorch MCMC Benchmark")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Composition
    composition = {"Sr": 32, "Ti": 8, "Fe": 24, "O": 84, "VO": 12}
    N = sum(composition.values())

    # Masks
    b_start = composition["Sr"]
    b_end = b_start + composition["Ti"] + composition["Fe"]
    o_start = b_end
    o_end = N

    b_site_mask = torch.zeros(N, dtype=torch.bool, device=device)
    b_site_mask[b_start:b_end] = True
    o_site_mask = torch.zeros(N, dtype=torch.bool, device=device)
    o_site_mask[o_start:o_end] = True

    for batch_size in [1000, 10000, 100000]:
        print(f"\n--- Batch size: {batch_size} ---")

        # Generate structures
        torch.cuda.synchronize() if device == 'cuda' else None
        start = time.perf_counter()
        structures = generate_structures_fast(
            batch_size,
            composition["Sr"], composition["Ti"], composition["Fe"],
            composition["O"], composition["VO"],
            device=device
        )
        torch.cuda.synchronize() if device == 'cuda' else None
        gen_time = time.perf_counter() - start
        print(f"  Generation: {gen_time*1000:.2f} ms")

        # Warm up swap
        for _ in range(3):
            _, _ = sample_sublattice_swap(
                structures, b_site_mask,
                ATOM_TYPES["Ti"], ATOM_TYPES["Fe"]
            )
        torch.cuda.synchronize() if device == 'cuda' else None

        # Benchmark swap
        times = []
        for _ in range(10):
            torch.cuda.synchronize() if device == 'cuda' else None
            start = time.perf_counter()
            _, _ = sample_sublattice_swap(
                structures, b_site_mask,
                ATOM_TYPES["Ti"], ATOM_TYPES["Fe"]
            )
            torch.cuda.synchronize() if device == 'cuda' else None
            times.append(time.perf_counter() - start)

        print(f"  Swap (B-site): {np.mean(times)*1000:.2f} ± {np.std(times)*1000:.2f} ms")
        print(f"  Throughput: {batch_size / np.mean(times):.0f} swaps/sec")


def run_mcmc_main(args):
    """Run single-temperature MCMC."""
    print("\n" + "=" * 60)
    print("Single Temperature MCMC")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Composition
    composition = {"Sr": 32, "Ti": 8, "Fe": 24, "O": 84, "VO": 12}
    N = sum(composition.values())

    # Masks
    b_start = composition["Sr"]
    b_end = b_start + composition["Ti"] + composition["Fe"]

    b_site_mask = torch.zeros(N, dtype=torch.bool, device=device)
    b_site_mask[b_start:b_end] = True
    o_site_mask = torch.zeros(N, dtype=torch.bool, device=device)
    o_site_mask[b_end:N] = True

    print(f"Composition: {composition}")
    print(f"Device: {device}")
    print(f"Temperature: {args.temperature}")

    # Generate structures
    print(f"\nGenerating {args.n_structures} structures...")
    structures = generate_structures_fast(
        args.n_structures,
        composition["Sr"], composition["Ti"], composition["Fe"],
        composition["O"], composition["VO"],
        device=device
    )

    # Dummy energy function for demo
    def dummy_energy_fn(atom_types):
        """Dummy energy: count Ti-Fe neighboring pairs (lower = better)."""
        # This is just a placeholder - real version would use trained model
        return torch.randn(atom_types.shape[0], device=atom_types.device)

    print("\n[Note: Using dummy energy function for demonstration]")
    print("For real optimization, load trained model checkpoint.\n")

    # Run MCMC
    best_types, best_energy, energy_hist, accept_hist = run_mcmc(
        structures,
        dummy_energy_fn,
        b_site_mask,
        o_site_mask,
        temperature=args.temperature,
        n_steps=args.n_steps,
    )

    print(f"\nFinal best mean energy: {best_energy.mean().item():.4f}")
    print(f"Final acceptance rate: {accept_hist[-1]:.3f}")


def main():
    parser = argparse.ArgumentParser(description="PyTorch MCMC for perovskites")
    parser.add_argument('--mode', type=str, default='mcmc',
                        choices=['mcmc', 'parallel_tempering', 'benchmark'])
    parser.add_argument('--ckpt', type=str, default='./checkpoints/best.pth')
    parser.add_argument('--seed', type=int, default=42)

    # MCMC parameters
    parser.add_argument('--n_structures', type=int, default=10000)
    parser.add_argument('--n_steps', type=int, default=1000)
    parser.add_argument('--temperature', type=float, default=0.1)

    # Parallel tempering
    parser.add_argument('--n_temps', type=int, default=100)
    parser.add_argument('--temp_min', type=float, default=0.01)
    parser.add_argument('--temp_max', type=float, default=1.0)

    args = parser.parse_args()
    torch.manual_seed(args.seed)

    if args.mode == 'mcmc':
        run_mcmc_main(args)
    elif args.mode == 'parallel_tempering':
        print("Parallel tempering: See run_parallel_tempering function")
    elif args.mode == 'benchmark':
        run_benchmark(args)


if __name__ == '__main__':
    main()
