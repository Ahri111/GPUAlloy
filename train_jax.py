"""
CGFormer Training with JAX/Flax

Usage:
    # Energy prediction training
    python train_jax.py --mode energy --data_dir ./STFO_data --epochs 100

    # Swap benchmark
    python train_jax.py --mode swap_benchmark

    # Inference only
    python train_jax.py --mode inference --ckpt ./checkpoints/params.pkl
"""

import argparse
import os
import pickle
import time
from typing import Optional, Tuple, Dict, Any
from functools import partial

import numpy as np

import jax
import jax.numpy as jnp
from jax import random, lax
import flax.linen as nn
from flax.training import train_state
import optax

from CGFormer_jax import CrystalGraphConvNet, CGFormerEncoder
from swap_utils_jax import (
    ATOM_TYPES,
    generate_structures,
    sample_sublattice_swap,
    sample_sublattice_swap_beam,
    apply_n_swaps,
    apply_n_swaps_both,
    log_prob_sublattice_swap,
    SwapResult, BeamResult,
)

print("="*60)
print(f"JAX version: {jax.__version__}")
print(f"Devices: {jax.devices()}")
print("="*60)


# =============================================================================
# Training State
# =============================================================================

class TrainState(train_state.TrainState):
    """Extended TrainState with batch stats for BatchNorm."""
    batch_stats: Optional[Dict[str, Any]] = None


def create_train_state(
    model: nn.Module,
    rng: random.PRNGKey,
    dummy_inputs: Dict[str, jnp.ndarray],
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
) -> TrainState:
    """Create initial training state."""
    variables = model.init(rng, **dummy_inputs, train=False)

    params = variables['params']
    batch_stats = variables.get('batch_stats', None)

    tx = optax.adamw(learning_rate=learning_rate, weight_decay=weight_decay)

    return TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
        batch_stats=batch_stats,
    )


# =============================================================================
# Loss Functions
# =============================================================================

def mse_loss(predictions: jnp.ndarray, targets: jnp.ndarray) -> jnp.ndarray:
    """Mean squared error loss."""
    return jnp.mean((predictions - targets) ** 2)


def mae_metric(predictions: jnp.ndarray, targets: jnp.ndarray) -> jnp.ndarray:
    """Mean absolute error metric."""
    return jnp.mean(jnp.abs(predictions - targets))


# =============================================================================
# Training Step
# =============================================================================

@partial(jax.jit, static_argnums=(0,))
def train_step(
    model: nn.Module,
    state: TrainState,
    batch: Tuple,
    normalizer_mean: float,
    normalizer_std: float,
):
    """Single training step."""
    (atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx), targets = batch

    def loss_fn(params):
        variables = {'params': params}
        if state.batch_stats is not None:
            variables['batch_stats'] = state.batch_stats

        outputs, mutated_vars = model.apply(
            variables,
            atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx,
            train=True,
            mutable=['batch_stats'] if state.batch_stats is not None else False,
        )

        # Normalize targets
        targets_normed = (targets - normalizer_mean) / normalizer_std
        loss = mse_loss(outputs, targets_normed)

        # Denormalize for MAE
        outputs_denorm = outputs * normalizer_std + normalizer_mean
        mae = mae_metric(outputs_denorm, targets)

        new_batch_stats = mutated_vars.get('batch_stats', None) if isinstance(mutated_vars, dict) else None

        return loss, (mae, new_batch_stats)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (mae, new_batch_stats)), grads = grad_fn(state.params)

    state = state.apply_gradients(grads=grads)
    if new_batch_stats is not None:
        state = state.replace(batch_stats=new_batch_stats)

    return state, loss, mae


@partial(jax.jit, static_argnums=(0,))
def eval_step(
    model: nn.Module,
    state: TrainState,
    batch: Tuple,
    normalizer_mean: float,
    normalizer_std: float,
):
    """Single evaluation step."""
    (atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx), targets = batch

    variables = {'params': state.params}
    if state.batch_stats is not None:
        variables['batch_stats'] = state.batch_stats

    outputs = model.apply(
        variables,
        atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx,
        train=False,
    )

    targets_normed = (targets - normalizer_mean) / normalizer_std
    loss = mse_loss(outputs, targets_normed)

    outputs_denorm = outputs * normalizer_std + normalizer_mean
    mae = mae_metric(outputs_denorm, targets)

    return loss, mae


# =============================================================================
# Inference
# =============================================================================

@partial(jax.jit, static_argnums=(0,))
def predict(
    model: nn.Module,
    params: Dict,
    batch_stats: Optional[Dict],
    atom_fea: jnp.ndarray,
    nbr_fea: jnp.ndarray,
    nbr_fea_idx: jnp.ndarray,
    crystal_atom_idx,
    normalizer_mean: float,
    normalizer_std: float,
):
    """Run inference and denormalize outputs."""
    variables = {'params': params}
    if batch_stats is not None:
        variables['batch_stats'] = batch_stats

    outputs = model.apply(
        variables,
        atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx,
        train=False,
    )

    # Denormalize
    return outputs * normalizer_std + normalizer_mean


# =============================================================================
# Swap Benchmark
# =============================================================================

def run_swap_benchmark(
    batch_size: int = 10000,
    n_swaps: int = 10000,
    composition: Dict[str, int] = None,
):
    """Benchmark swap operations."""
    if composition is None:
        composition = {"Sr": 32, "Ti": 8, "Fe": 24, "O": 84, "VO": 12}

    N = sum(composition.values())
    n_B = composition["Ti"] + composition["Fe"]

    sr_end = composition["Sr"]
    b_end = sr_end + n_B
    b_site_idx = tuple(range(sr_end, b_end))
    o_site_idx = tuple(range(b_end, N))

    print(f"\n{'='*60}")
    print("SWAP BENCHMARK")
    print(f"{'='*60}")
    print(f"Batch size: {batch_size}")
    print(f"N swaps: {n_swaps}")
    print(f"Atoms per structure: {N}")
    print(f"B-site (Ti↔Fe): {len(b_site_idx)} positions")
    print(f"O-site (O↔VO): {len(o_site_idx)} positions")

    # Generate structures
    key = random.PRNGKey(42)
    key, subkey = random.split(key)

    structures = generate_structures(
        subkey, batch_size,
        composition["Sr"], composition["Ti"], composition["Fe"],
        composition["O"], composition["VO"]
    )
    structures.block_until_ready()
    print(f"\nGenerated {batch_size} structures: {structures.shape}")

    # Scores
    key, subkey = random.split(key)
    scores = random.normal(subkey, structures.shape)

    def benchmark(name, fn, n_warmup=3, n_trials=5):
        for _ in range(n_warmup):
            out = fn()
            if hasattr(out, 'block_until_ready'):
                out.block_until_ready()
            elif isinstance(out, tuple):
                out[0].block_until_ready()

        times = []
        for _ in range(n_trials):
            start = time.perf_counter()
            out = fn()
            if hasattr(out, 'block_until_ready'):
                out.block_until_ready()
            elif isinstance(out, tuple):
                out[0].block_until_ready()
            times.append(time.perf_counter() - start)

        avg = sum(times) / len(times)
        print(f"{name:40s}: {avg*1000:10.2f} ms")
        return avg

    # Single swap
    print(f"\n[1] sample_sublattice_swap")
    key, subkey = random.split(key)
    benchmark(
        "B-site swap",
        lambda: sample_sublattice_swap(subkey, structures, b_site_idx, 1, 2, scores)
    )

    # N swaps - B-site only
    print(f"\n[2] apply_n_swaps B-site only ({n_swaps}x)")
    key, subkey = random.split(key)
    benchmark(
        "B-site only",
        lambda: apply_n_swaps(subkey, structures, scores, b_site_idx, 1, 2, n_swaps),
        n_warmup=2, n_trials=3
    )

    # N swaps - BOTH mode
    print(f"\n[3] apply_n_swaps_both ({n_swaps}x)")
    key, subkey = random.split(key)
    benchmark(
        "BOTH (B+O random)",
        lambda: apply_n_swaps_both(
            subkey, structures, scores,
            b_site_idx, o_site_idx,
            ATOM_TYPES["Ti"], ATOM_TYPES["Fe"],
            ATOM_TYPES["O"], ATOM_TYPES["VO"],
            n_swaps
        ),
        n_warmup=2, n_trials=3
    )

    # Beam search
    print(f"\n[4] beam_search (k=4)")
    benchmark(
        "B-site beam",
        lambda: sample_sublattice_swap_beam(structures, b_site_idx, 1, 2, scores, 4)
    )

    print(f"\n{'='*60}")
    print("Benchmark completed!")


# =============================================================================
# Save/Load
# =============================================================================

def save_checkpoint(path: str, state: TrainState, normalizer: Dict):
    """Save checkpoint."""
    checkpoint = {
        'params': state.params,
        'batch_stats': state.batch_stats,
        'opt_state': state.opt_state,
        'step': state.step,
        'normalizer': normalizer,
    }
    with open(path, 'wb') as f:
        pickle.dump(checkpoint, f)
    print(f"Saved checkpoint to {path}")


def load_checkpoint(path: str) -> Dict:
    """Load checkpoint."""
    with open(path, 'rb') as f:
        checkpoint = pickle.load(f)
    print(f"Loaded checkpoint from {path}")
    return checkpoint


# =============================================================================
# Example Training Loop (without DataLoader)
# =============================================================================

def example_training_loop():
    """Example showing JAX training loop structure."""
    print("\n" + "="*60)
    print("Example JAX Training Loop (Structure Only)")
    print("="*60)

    # Model
    model = CrystalGraphConvNet(
        orig_atom_fea_len=92,
        nbr_fea_len=41,
        atom_fea_len=64,
        n_conv=3,
        h_fea_len=128,
        n_h=1,
        graphormer_layers=1,
        num_heads=4,
    )

    # Dummy inputs for initialization
    rng = random.PRNGKey(0)
    dummy_atom_fea = jnp.zeros((10, 92))
    dummy_nbr_fea = jnp.zeros((10, 12, 41))
    dummy_nbr_fea_idx = jnp.zeros((10, 12), dtype=jnp.int32)
    dummy_crystal_atom_idx = [jnp.arange(10)]

    # Initialize
    state = create_train_state(
        model, rng,
        {
            'atom_fea': dummy_atom_fea,
            'nbr_fea': dummy_nbr_fea,
            'nbr_fea_idx': dummy_nbr_fea_idx,
            'crystal_atom_idx': dummy_crystal_atom_idx,
        },
        learning_rate=1e-3,
    )

    print(f"Model initialized!")
    print(f"  Params: {sum(p.size for p in jax.tree_util.tree_leaves(state.params)):,}")

    # Training loop structure
    print("""
Training loop structure:

    for epoch in range(num_epochs):
        # Training
        for batch in train_data:
            state, loss, mae = train_step(model, state, batch, mean, std)

        # Validation
        val_losses = []
        for batch in val_data:
            loss, mae = eval_step(model, state, batch, mean, std)
            val_losses.append(mae)

        val_mae = jnp.mean(jnp.array(val_losses))

        # Checkpoint
        if val_mae < best_mae:
            save_checkpoint('best.pkl', state, normalizer)
            best_mae = val_mae
    """)

    return state


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='CGFormer JAX Training')

    parser.add_argument('--mode', type=str, default='swap_benchmark',
                        choices=['energy', 'swap_benchmark', 'inference', 'example'],
                        help='Mode')
    parser.add_argument('--data_dir', type=str, default='./STFO_data')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--ckpt_dir', type=str, default='./checkpoints_jax')

    # Swap benchmark
    parser.add_argument('--swap_batch', type=int, default=10000)
    parser.add_argument('--n_swaps', type=int, default=10000)

    args = parser.parse_args()

    os.makedirs(args.ckpt_dir, exist_ok=True)

    if args.mode == 'swap_benchmark':
        run_swap_benchmark(
            batch_size=args.swap_batch,
            n_swaps=args.n_swaps,
        )
    elif args.mode == 'example':
        example_training_loop()
    elif args.mode == 'energy':
        print("Full JAX energy training requires data loading integration.")
        print("See example_training_loop() for structure.")
        example_training_loop()
    elif args.mode == 'inference':
        if args.ckpt is None:
            raise ValueError("--ckpt required for inference mode")
        checkpoint = load_checkpoint(args.ckpt)
        print(f"Loaded params with {len(checkpoint['params'])} modules")


if __name__ == '__main__':
    main()
