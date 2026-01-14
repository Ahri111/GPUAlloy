"""
MCMC Optimization for Perovskite Structures with JAX

스왑 → 에너지 평가 → MCMC → 반복 파이프라인

핵심: 스왑 시 atom_types만 변경, edge_index는 유지됨 (결정 구조 불변)

Usage:
    # Single temperature MCMC
    python mcmc_jax.py --ckpt ./checkpoints_jax/best.pkl --n_structures 10000 --temperature 0.1

    # Parallel tempering (multiple temperatures)
    python mcmc_jax.py --mode parallel_tempering --ckpt ./checkpoints_jax/best.pkl \
        --n_structures 100000 --n_temps 100 --temp_min 0.01 --temp_max 1.0

    # Benchmark MCMC speed
    python mcmc_jax.py --mode benchmark
"""

import argparse
import pickle
import time
from typing import Optional, Dict, Tuple, List
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
from jax import random, lax
import flax.linen as nn

from CGFormer_jax import CrystalGraphConvNet
from swap_utils_jax import (
    ATOM_TYPES, generate_structures,
    apply_n_swaps, apply_n_swaps_both,
    sample_sublattice_swap,
)

print("=" * 60)
print(f"JAX MCMC Pipeline")
print(f"JAX version: {jax.__version__}")
print(f"Devices: {jax.devices()}")
print("=" * 60)


# =============================================================================
# Atom Feature Lookup (핵심: 스왑 후 빠른 feature 업데이트)
# =============================================================================

def create_atom_feature_table(atom_init_path: str = 'atom_init.json') -> jnp.ndarray:
    """Load atom feature lookup table.

    ABO3 perovskite의 원소들에 대한 feature table 생성.
    Returns: (max_atomic_num + 1, feature_dim) array
    """
    import json
    with open(atom_init_path, 'r') as f:
        atom_features = json.load(f)

    # Find max atomic number and feature dim
    max_z = max(int(k) for k in atom_features.keys())
    fea_dim = len(next(iter(atom_features.values())))

    # Create lookup table (index 0 for vacancy/dummy)
    table = np.zeros((max_z + 2, fea_dim), dtype=np.float32)
    for z_str, fea in atom_features.items():
        table[int(z_str)] = fea

    return jnp.array(table)


# Atom type to atomic number mapping (ABO3 perovskite)
ATOMIC_NUMBERS = {
    "Sr": 38,
    "Ti": 22,
    "Fe": 26,
    "O": 8,
    "VO": 0,  # Vacancy = 0
}


@jax.jit
def atom_types_to_features(
    atom_types: jnp.ndarray,
    feature_table: jnp.ndarray,
    type_to_z: jnp.ndarray,
) -> jnp.ndarray:
    """Convert atom type indices to atom features.

    Args:
        atom_types: (batch_size, n_atoms) - atom type indices (0=Sr, 1=Ti, 2=Fe, 3=O, 4=VO)
        feature_table: (max_z + 1, fea_dim) - lookup table
        type_to_z: (5,) - mapping from type index to atomic number

    Returns:
        atom_fea: (batch_size, n_atoms, fea_dim)
    """
    # Map type indices to atomic numbers
    atomic_nums = type_to_z[atom_types]  # (batch_size, n_atoms)

    # Look up features
    atom_fea = feature_table[atomic_nums]  # (batch_size, n_atoms, fea_dim)

    return atom_fea


# =============================================================================
# Energy Evaluation (모델 추론)
# =============================================================================

def create_energy_fn(
    model: nn.Module,
    params: Dict,
    batch_stats: Optional[Dict],
    normalizer_mean: float,
    normalizer_std: float,
):
    """Create JIT-compiled energy evaluation function.

    Returns:
        energy_fn: (atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx) -> energies
    """

    @jax.jit
    def energy_fn(atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx):
        variables = {'params': params}
        if batch_stats is not None:
            variables['batch_stats'] = batch_stats

        # Model forward pass
        outputs = model.apply(
            variables,
            atom_fea,
            nbr_fea,
            nbr_fea_idx,
            crystal_atom_idx,
            train=False
        )

        # Denormalize
        energies = outputs.squeeze(-1) * normalizer_std + normalizer_mean
        return energies

    return energy_fn


# =============================================================================
# MCMC Step (Metropolis-Hastings)
# =============================================================================

@partial(jax.jit, static_argnums=(7, 8, 9, 10, 11, 12))
def mcmc_step(
    key: random.PRNGKey,
    atom_types: jnp.ndarray,
    current_energy: jnp.ndarray,
    scores: jnp.ndarray,  # For guided swap (can be zeros for random)
    temperature: jnp.ndarray,  # (batch_size,) or scalar
    feature_table: jnp.ndarray,
    type_to_z: jnp.ndarray,
    b_site_indices: Tuple[int, ...],
    o_site_indices: Tuple[int, ...],
    type_ti: int,
    type_fe: int,
    type_o: int,
    type_vo: int,
    energy_fn,
    nbr_fea: jnp.ndarray,
    nbr_fea_idx: jnp.ndarray,
    crystal_atom_idx: List[jnp.ndarray],
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Single MCMC step with Metropolis-Hastings acceptance.

    Args:
        key: Random key
        atom_types: (batch_size, n_atoms) current atom types
        current_energy: (batch_size,) current energies
        scores: (batch_size, n_atoms) scores for guided swap
        temperature: (batch_size,) or scalar temperature
        ... (other args)

    Returns:
        new_atom_types: Updated atom types
        new_energy: Updated energies
        accepted: (batch_size,) boolean mask of accepted moves
        accept_prob: (batch_size,) acceptance probabilities
    """
    batch_size = atom_types.shape[0]
    key, swap_key, accept_key = random.split(key, 3)

    # 1. Propose swap (apply_n_swaps_both for B-site and O-site)
    proposed_types = apply_n_swaps_both(
        swap_key, atom_types, scores,
        b_site_indices, o_site_indices,
        type_ti, type_fe, type_o, type_vo,
        n_swaps=1  # Single swap per step
    )

    # 2. Convert proposed types to features
    proposed_fea = atom_types_to_features(proposed_types, feature_table, type_to_z)

    # 3. Flatten for model input (all structures concatenated)
    # Note: This assumes fixed structure size for JIT
    n_atoms = atom_types.shape[1]
    flat_fea = proposed_fea.reshape(-1, proposed_fea.shape[-1])

    # 4. Evaluate proposed energy
    proposed_energy = energy_fn(flat_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx)

    # 5. Metropolis-Hastings acceptance
    delta_E = proposed_energy - current_energy

    # Accept probability: min(1, exp(-dE/T))
    # For minimization: accept if E_new < E_old, or with prob exp(-dE/T) if E_new > E_old
    accept_prob = jnp.minimum(1.0, jnp.exp(-delta_E / temperature))

    # Random acceptance
    u = random.uniform(accept_key, (batch_size,))
    accepted = u < accept_prob

    # 6. Update based on acceptance
    new_atom_types = jnp.where(accepted[:, None], proposed_types, atom_types)
    new_energy = jnp.where(accepted, proposed_energy, current_energy)

    return new_atom_types, new_energy, accepted, accept_prob


# =============================================================================
# MCMC Loop with lax.scan (효율적인 반복)
# =============================================================================

def create_mcmc_loop(
    energy_fn,
    feature_table: jnp.ndarray,
    type_to_z: jnp.ndarray,
    nbr_fea: jnp.ndarray,
    nbr_fea_idx: jnp.ndarray,
    crystal_atom_idx: List[jnp.ndarray],
    b_site_indices: Tuple[int, ...],
    o_site_indices: Tuple[int, ...],
    composition: Dict[str, int],
):
    """Create MCMC loop function using lax.scan.

    lax.scan을 사용하여 효율적인 MCMC 반복 구현.
    """
    type_ti = ATOM_TYPES["Ti"]
    type_fe = ATOM_TYPES["Fe"]
    type_o = ATOM_TYPES["O"]
    type_vo = ATOM_TYPES["VO"]

    def mcmc_body(carry, key):
        """Single MCMC step for lax.scan."""
        atom_types, energy, total_accepted = carry

        # Use zeros for scores (random swap)
        scores = jnp.zeros_like(atom_types, dtype=jnp.float32)
        temperature = carry[3] if len(carry) > 3 else 0.1

        new_types, new_energy, accepted, _ = mcmc_step(
            key, atom_types, energy, scores, temperature,
            feature_table, type_to_z,
            b_site_indices, o_site_indices,
            type_ti, type_fe, type_o, type_vo,
            energy_fn, nbr_fea, nbr_fea_idx, crystal_atom_idx,
        )

        new_accepted = total_accepted + accepted.astype(jnp.int32)
        return (new_types, new_energy, new_accepted), (new_energy, accepted)

    def run_mcmc(
        key: random.PRNGKey,
        init_atom_types: jnp.ndarray,
        init_energy: jnp.ndarray,
        temperature: jnp.ndarray,
        n_steps: int,
    ):
        """Run MCMC for n_steps.

        Args:
            key: Random key
            init_atom_types: (batch_size, n_atoms)
            init_energy: (batch_size,)
            temperature: (batch_size,) or scalar
            n_steps: Number of MCMC steps

        Returns:
            final_atom_types: (batch_size, n_atoms)
            final_energy: (batch_size,)
            energy_history: (n_steps, batch_size)
            accept_history: (n_steps, batch_size)
            total_accepted: (batch_size,)
        """
        batch_size = init_atom_types.shape[0]
        keys = random.split(key, n_steps)

        init_carry = (init_atom_types, init_energy, jnp.zeros(batch_size, dtype=jnp.int32), temperature)

        # Run with lax.scan
        def scan_body(carry, key):
            atom_types, energy, total_accepted, temp = carry
            scores = jnp.zeros_like(atom_types, dtype=jnp.float32)

            new_types, new_energy, accepted, _ = mcmc_step(
                key, atom_types, energy, scores, temp,
                feature_table, type_to_z,
                b_site_indices, o_site_indices,
                type_ti, type_fe, type_o, type_vo,
                energy_fn, nbr_fea, nbr_fea_idx, crystal_atom_idx,
            )

            new_accepted = total_accepted + accepted.astype(jnp.int32)
            return (new_types, new_energy, new_accepted, temp), (new_energy, accepted)

        final_carry, (energy_history, accept_history) = lax.scan(
            scan_body, init_carry, keys
        )

        final_types, final_energy, total_accepted, _ = final_carry

        return final_types, final_energy, energy_history, accept_history, total_accepted

    return run_mcmc


# =============================================================================
# Parallel Tempering (다중 온도)
# =============================================================================

def create_parallel_tempering(
    energy_fn,
    feature_table: jnp.ndarray,
    type_to_z: jnp.ndarray,
    nbr_fea: jnp.ndarray,
    nbr_fea_idx: jnp.ndarray,
    crystal_atom_idx: List[jnp.ndarray],
    b_site_indices: Tuple[int, ...],
    o_site_indices: Tuple[int, ...],
    composition: Dict[str, int],
    n_temps: int,
    temp_min: float,
    temp_max: float,
):
    """Create parallel tempering function.

    100,000개 구조를 1000개씩 100개 온도에 배정하여 병렬 처리.
    """
    # Create temperature ladder (log scale)
    temperatures = jnp.logspace(jnp.log10(temp_min), jnp.log10(temp_max), n_temps)

    type_ti = ATOM_TYPES["Ti"]
    type_fe = ATOM_TYPES["Fe"]
    type_o = ATOM_TYPES["O"]
    type_vo = ATOM_TYPES["VO"]

    def parallel_tempering_step(
        key: random.PRNGKey,
        atom_types: jnp.ndarray,  # (n_temps, structures_per_temp, n_atoms)
        energies: jnp.ndarray,    # (n_temps, structures_per_temp)
    ):
        """Single parallel tempering step.

        1. MCMC step at each temperature
        2. Attempt replica exchange between adjacent temperatures
        """
        n_temps_actual, structures_per_temp, n_atoms = atom_types.shape

        key, mcmc_key, exchange_key = random.split(key, 3)

        # 1. MCMC step at each temperature
        # Reshape for batch processing
        flat_types = atom_types.reshape(-1, n_atoms)
        flat_energies = energies.reshape(-1)

        # Temperature for each structure
        temp_per_struct = jnp.repeat(temperatures, structures_per_temp)

        scores = jnp.zeros_like(flat_types, dtype=jnp.float32)

        new_types, new_energies, accepted, _ = mcmc_step(
            mcmc_key, flat_types, flat_energies, scores, temp_per_struct,
            feature_table, type_to_z,
            b_site_indices, o_site_indices,
            type_ti, type_fe, type_o, type_vo,
            energy_fn, nbr_fea, nbr_fea_idx, crystal_atom_idx,
        )

        # Reshape back
        new_types = new_types.reshape(n_temps_actual, structures_per_temp, n_atoms)
        new_energies = new_energies.reshape(n_temps_actual, structures_per_temp)

        # 2. Replica exchange (between adjacent temperatures)
        # For each pair of adjacent temperatures, attempt swap
        exchange_keys = random.split(exchange_key, n_temps_actual - 1)

        def attempt_exchange(i, state):
            types, energies = state
            key_i = exchange_keys[i]

            # Energy difference for Metropolis criterion
            # delta = (1/T_i - 1/T_{i+1}) * (E_i - E_{i+1})
            beta_diff = 1.0 / temperatures[i] - 1.0 / temperatures[i + 1]

            # For each structure pair
            for j in range(structures_per_temp):
                delta = beta_diff * (energies[i, j] - energies[i + 1, j])
                accept_prob = jnp.minimum(1.0, jnp.exp(delta))
                u = random.uniform(random.fold_in(key_i, j))

                if u < accept_prob:
                    # Swap
                    types = types.at[i, j].set(types[i + 1, j])
                    types = types.at[i + 1, j].set(types[i, j])
                    energies = energies.at[i, j].set(energies[i + 1, j])
                    energies = energies.at[i + 1, j].set(energies[i, j])

            return (types, energies)

        # Note: For full JIT, would need lax.fori_loop
        # For now, skip exchange in JIT version (can add later)

        return new_types, new_energies, accepted.reshape(n_temps_actual, structures_per_temp)

    def run_parallel_tempering(
        key: random.PRNGKey,
        init_atom_types: jnp.ndarray,  # (n_temps, structures_per_temp, n_atoms)
        n_steps: int,
        exchange_every: int = 10,
    ):
        """Run parallel tempering for n_steps."""
        n_temps_actual, structures_per_temp, n_atoms = init_atom_types.shape

        # Initial energy evaluation
        flat_types = init_atom_types.reshape(-1, n_atoms)
        flat_fea = atom_types_to_features(flat_types, feature_table, type_to_z)
        flat_fea = flat_fea.reshape(-1, flat_fea.shape[-1])

        init_energies = energy_fn(flat_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx)
        init_energies = init_energies.reshape(n_temps_actual, structures_per_temp)

        # Run MCMC
        atom_types = init_atom_types
        energies = init_energies

        best_types = atom_types[0]  # Lowest temperature
        best_energies = energies[0]

        energy_history = []

        for step in range(n_steps):
            key, step_key = random.split(key)
            atom_types, energies, _ = parallel_tempering_step(
                step_key, atom_types, energies
            )

            # Track best (lowest temperature)
            improved = energies[0] < best_energies
            best_types = jnp.where(improved[:, None], atom_types[0], best_types)
            best_energies = jnp.where(improved, energies[0], best_energies)

            if step % 100 == 0:
                energy_history.append(float(jnp.mean(best_energies)))
                print(f"Step {step}: Best mean energy = {energy_history[-1]:.4f}")

        return best_types, best_energies, energy_history

    return run_parallel_tempering, temperatures


# =============================================================================
# Main Functions
# =============================================================================

def run_mcmc(args):
    """Run single-temperature MCMC optimization."""
    print("\n" + "=" * 60)
    print("Single Temperature MCMC")
    print("=" * 60)

    # Load checkpoint
    with open(args.ckpt, 'rb') as f:
        ckpt = pickle.load(f)
    print(f"Loaded checkpoint: {args.ckpt}")

    normalizer_mean = ckpt['normalizer']['mean']
    normalizer_std = ckpt['normalizer']['std']

    # Composition (ABO3 perovskite)
    composition = {"Sr": 32, "Ti": 8, "Fe": 24, "O": 84, "VO": 12}
    N = sum(composition.values())

    # Site indices
    sr_end = composition["Sr"]
    b_end = sr_end + composition["Ti"] + composition["Fe"]
    b_site_indices = tuple(range(sr_end, b_end))
    o_site_indices = tuple(range(b_end, N))

    print(f"Composition: {composition}")
    print(f"Total atoms: {N}")
    print(f"B-site indices: {b_site_indices[0]}..{b_site_indices[-1]}")
    print(f"O-site indices: {o_site_indices[0]}..{o_site_indices[-1]}")

    # Generate initial structures
    key = random.PRNGKey(args.seed)
    key, init_key = random.split(key)

    print(f"\nGenerating {args.n_structures} random structures...")
    init_types = generate_structures(
        init_key, args.n_structures,
        composition["Sr"], composition["Ti"], composition["Fe"],
        composition["O"], composition["VO"]
    )
    print(f"Initial structures shape: {init_types.shape}")

    # Load atom features
    try:
        feature_table = create_atom_feature_table(args.atom_init)
    except FileNotFoundError:
        print(f"Warning: {args.atom_init} not found, using random features")
        feature_table = jnp.array(np.random.randn(100, 92).astype(np.float32))

    type_to_z = jnp.array([
        ATOMIC_NUMBERS["Sr"],
        ATOMIC_NUMBERS["Ti"],
        ATOMIC_NUMBERS["Fe"],
        ATOMIC_NUMBERS["O"],
        ATOMIC_NUMBERS["VO"],
    ])

    # Create dummy graph structure (fixed for all structures)
    # In real use, this would come from the crystal structure
    print("\nNote: Using dummy graph structure for demonstration.")
    print("In production, load actual nbr_fea and nbr_fea_idx from data.")

    # For now, just run benchmark without full model
    print("\n" + "=" * 60)
    print("MCMC Benchmark (swap operations only)")
    print("=" * 60)

    temperature = args.temperature
    n_steps = args.n_steps

    # Benchmark swap speed
    key, bench_key = random.split(key)
    scores = jnp.zeros_like(init_types, dtype=jnp.float32)

    # Warm up
    for _ in range(3):
        new_types = apply_n_swaps_both(
            bench_key, init_types, scores,
            b_site_indices, o_site_indices,
            ATOM_TYPES["Ti"], ATOM_TYPES["Fe"],
            ATOM_TYPES["O"], ATOM_TYPES["VO"],
            n_swaps=1
        )
        new_types.block_until_ready()

    # Benchmark
    times = []
    for i in range(10):
        key, step_key = random.split(key)
        start = time.perf_counter()
        new_types = apply_n_swaps_both(
            step_key, init_types, scores,
            b_site_indices, o_site_indices,
            ATOM_TYPES["Ti"], ATOM_TYPES["Fe"],
            ATOM_TYPES["O"], ATOM_TYPES["VO"],
            n_swaps=1
        )
        new_types.block_until_ready()
        times.append(time.perf_counter() - start)

    print(f"\nSwap benchmark ({args.n_structures} structures):")
    print(f"  Mean: {np.mean(times)*1000:.2f} ms")
    print(f"  Std:  {np.std(times)*1000:.2f} ms")
    print(f"  Throughput: {args.n_structures / np.mean(times):.0f} swaps/sec")


def run_benchmark(args):
    """Benchmark MCMC operations."""
    print("\n" + "=" * 60)
    print("MCMC Benchmark")
    print("=" * 60)

    # Composition
    composition = {"Sr": 32, "Ti": 8, "Fe": 24, "O": 84, "VO": 12}
    N = sum(composition.values())

    sr_end = composition["Sr"]
    b_end = sr_end + composition["Ti"] + composition["Fe"]
    b_site_indices = tuple(range(sr_end, b_end))
    o_site_indices = tuple(range(b_end, N))

    key = random.PRNGKey(42)

    for batch_size in [1000, 10000, 100000]:
        key, init_key = random.split(key)
        structures = generate_structures(
            init_key, batch_size,
            composition["Sr"], composition["Ti"], composition["Fe"],
            composition["O"], composition["VO"]
        )
        structures.block_until_ready()

        scores = jnp.zeros_like(structures, dtype=jnp.float32)

        # Warm up
        for _ in range(3):
            key, step_key = random.split(key)
            _ = apply_n_swaps_both(
                step_key, structures, scores,
                b_site_indices, o_site_indices,
                ATOM_TYPES["Ti"], ATOM_TYPES["Fe"],
                ATOM_TYPES["O"], ATOM_TYPES["VO"],
                n_swaps=1
            )
            _.block_until_ready()

        # Benchmark
        times = []
        for _ in range(10):
            key, step_key = random.split(key)
            start = time.perf_counter()
            new = apply_n_swaps_both(
                step_key, structures, scores,
                b_site_indices, o_site_indices,
                ATOM_TYPES["Ti"], ATOM_TYPES["Fe"],
                ATOM_TYPES["O"], ATOM_TYPES["VO"],
                n_swaps=1
            )
            new.block_until_ready()
            times.append(time.perf_counter() - start)

        print(f"\nBatch size: {batch_size}")
        print(f"  Mean: {np.mean(times)*1000:.2f} ms")
        print(f"  Throughput: {batch_size / np.mean(times):.0f} swaps/sec")


def main():
    parser = argparse.ArgumentParser(description="MCMC optimization for perovskites")
    parser.add_argument('--mode', type=str, default='mcmc',
                        choices=['mcmc', 'parallel_tempering', 'benchmark'])
    parser.add_argument('--ckpt', type=str, default='./checkpoints_jax/best.pkl')
    parser.add_argument('--atom_init', type=str, default='atom_init.json')
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

    if args.mode == 'mcmc':
        run_mcmc(args)
    elif args.mode == 'parallel_tempering':
        print("Parallel tempering: Coming soon!")
        # run_parallel_tempering(args)
    elif args.mode == 'benchmark':
        run_benchmark(args)


if __name__ == '__main__':
    main()
