"""
CGFormer Training with JAX/Flax

JAX 모델 저장 형식: pickle (.pkl)

가장 많이 쓰는 방식: PyTorch DataLoader + JAX 변환

Usage:
    # Energy prediction training
    python train_jax.py --mode train --data_dir ./STFO_data --epochs 100

    # Inference (예측)
    python train_jax.py --mode predict --ckpt ./checkpoints_jax/best.pkl --data_dir ./STFO_data

    # Swap benchmark
    python train_jax.py --mode swap_benchmark
"""

import argparse
import os
import pickle
import time
from typing import Optional, Dict, Any, List
from functools import partial

import numpy as np

import jax
import jax.numpy as jnp
from jax import random
import flax.linen as nn
from flax.training import train_state
import optax

# PyTorch for DataLoader (가장 많이 쓰는 방식)
import torch
from torch.utils.data import DataLoader

from CGFormer_jax import CrystalGraphConvNet
from data import CIFData, collate_pool, get_train_val_test_loader
from swap_utils_jax import (
    ATOM_TYPES, generate_structures,
    sample_sublattice_swap, sample_sublattice_swap_beam,
    apply_n_swaps, apply_n_swaps_both,
)

print("="*60)
print(f"JAX version: {jax.__version__}")
print(f"Devices: {jax.devices()}")
print("="*60)


# =============================================================================
# PyTorch -> JAX 변환
# =============================================================================

def torch_to_jax(x):
    """Convert PyTorch tensor to JAX array."""
    if isinstance(x, torch.Tensor):
        return jnp.array(x.detach().cpu().numpy())
    return x


def convert_batch(batch):
    """Convert PyTorch batch to JAX format."""
    (atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx), target, cif_ids = batch
    return {
        'atom_fea': torch_to_jax(atom_fea),
        'nbr_fea': torch_to_jax(nbr_fea),
        'nbr_fea_idx': torch_to_jax(nbr_fea_idx),
        'crystal_atom_idx': [torch_to_jax(idx) for idx in crystal_atom_idx],
        'target': torch_to_jax(target),
        'cif_ids': cif_ids,
    }


# =============================================================================
# Normalizer
# =============================================================================

class Normalizer:
    """Normalize targets to zero mean and unit variance."""

    def __init__(self, data=None):
        if data is not None:
            self.mean = float(np.mean(data))
            self.std = float(np.std(data))
        else:
            self.mean = 0.0
            self.std = 1.0

    def norm(self, x):
        return (x - self.mean) / self.std

    def denorm(self, x):
        return x * self.std + self.mean

    def state_dict(self):
        return {'mean': self.mean, 'std': self.std}

    def load_state_dict(self, d):
        self.mean = d['mean']
        self.std = d['std']


# =============================================================================
# Training State
# =============================================================================

class TrainState(train_state.TrainState):
    """Extended TrainState with batch stats."""
    batch_stats: Optional[Dict] = None


def create_train_state(
    model: nn.Module,
    rng: random.PRNGKey,
    dummy_batch: Dict,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
) -> TrainState:
    """Create initial training state."""
    variables = model.init(
        rng,
        dummy_batch['atom_fea'],
        dummy_batch['nbr_fea'],
        dummy_batch['nbr_fea_idx'],
        dummy_batch['crystal_atom_idx'],
        train=False
    )

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
# Training / Eval Steps
# =============================================================================

def train_step(model, state, batch, normalizer):
    """Single training step."""

    def loss_fn(params):
        variables = {'params': params}
        if state.batch_stats is not None:
            variables['batch_stats'] = state.batch_stats

        if state.batch_stats is not None:
            outputs, mutated = model.apply(
                variables,
                batch['atom_fea'],
                batch['nbr_fea'],
                batch['nbr_fea_idx'],
                batch['crystal_atom_idx'],
                train=True,
                mutable=['batch_stats']
            )
            new_batch_stats = mutated['batch_stats']
        else:
            outputs = model.apply(
                variables,
                batch['atom_fea'],
                batch['nbr_fea'],
                batch['nbr_fea_idx'],
                batch['crystal_atom_idx'],
                train=True
            )
            new_batch_stats = None

        targets = batch['target'].squeeze(-1)
        targets_norm = normalizer.norm(targets)
        loss = jnp.mean((outputs.squeeze(-1) - targets_norm) ** 2)

        return loss, (outputs, new_batch_stats)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (outputs, new_batch_stats)), grads = grad_fn(state.params)

    state = state.apply_gradients(grads=grads)
    if new_batch_stats is not None:
        state = state.replace(batch_stats=new_batch_stats)

    # MAE
    preds = normalizer.denorm(outputs.squeeze(-1))
    targets = batch['target'].squeeze(-1)
    mae = jnp.mean(jnp.abs(preds - targets))

    return state, loss, mae


def eval_step(model, state, batch, normalizer):
    """Single evaluation step."""
    variables = {'params': state.params}
    if state.batch_stats is not None:
        variables['batch_stats'] = state.batch_stats

    outputs = model.apply(
        variables,
        batch['atom_fea'],
        batch['nbr_fea'],
        batch['nbr_fea_idx'],
        batch['crystal_atom_idx'],
        train=False
    )

    targets = batch['target'].squeeze(-1)
    targets_norm = normalizer.norm(targets)
    loss = jnp.mean((outputs.squeeze(-1) - targets_norm) ** 2)

    preds = normalizer.denorm(outputs.squeeze(-1))
    mae = jnp.mean(jnp.abs(preds - targets))

    return loss, mae, preds


# =============================================================================
# Save / Load Checkpoint
# =============================================================================

def save_checkpoint(path: str, state: TrainState, normalizer: Normalizer, epoch: int):
    """Save checkpoint as pickle."""
    checkpoint = {
        'params': jax.device_get(state.params),
        'batch_stats': jax.device_get(state.batch_stats) if state.batch_stats else None,
        'opt_state': jax.device_get(state.opt_state),
        'step': int(state.step),
        'normalizer': normalizer.state_dict(),
        'epoch': epoch,
    }
    with open(path, 'wb') as f:
        pickle.dump(checkpoint, f)
    print(f"Saved: {path}")


def load_checkpoint(path: str) -> Dict:
    """Load checkpoint from pickle."""
    with open(path, 'rb') as f:
        checkpoint = pickle.load(f)
    print(f"Loaded: {path}")
    return checkpoint


# =============================================================================
# Training Loop
# =============================================================================

def train(args):
    """Full training loop using PyTorch DataLoader + JAX."""
    print("\n" + "="*60)
    print("JAX Training (PyTorch DataLoader)")
    print("="*60)

    # Dataset (PyTorch)
    print(f"\nLoading data from: {args.data_dir}")
    dataset = CIFData(
        root_dir=args.data_dir,
        max_num_nbr=12,
        radius=8.0,
    )
    print(f"Dataset size: {len(dataset)}")

    # DataLoaders (PyTorch - 멀티프로세스 지원)
    train_loader, val_loader, test_loader = get_train_val_test_loader(
        dataset=dataset,
        collate_fn=collate_pool,
        batch_size=args.batch_size,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        return_test=True,
        num_workers=args.num_workers,
        pin_memory=False,
        train_size=None,
        val_size=None,
        test_size=None,
    )
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Normalizer (PyTorch로 빠르게 계산)
    print("Computing normalizer...")
    train_targets = []
    for batch in train_loader:
        _, target, _ = batch
        train_targets.append(target)
    train_targets = torch.cat(train_targets, dim=0).numpy()
    normalizer = Normalizer(train_targets)
    print(f"Target mean: {normalizer.mean:.4f}, std: {normalizer.std:.4f}")

    # Sample batch for model init
    sample_pt_batch = next(iter(train_loader))
    sample_batch = convert_batch(sample_pt_batch)

    # Model
    model = CrystalGraphConvNet(
        orig_atom_fea_len=sample_batch['atom_fea'].shape[-1],
        nbr_fea_len=sample_batch['nbr_fea'].shape[-1],
        atom_fea_len=args.atom_fea_len,
        n_conv=args.n_conv,
        h_fea_len=args.h_fea_len,
        n_h=args.n_h,
        graphormer_layers=args.graphormer_layers,
        num_heads=args.num_heads,
    )

    # Initialize
    rng = random.PRNGKey(args.seed)
    rng, init_rng = random.split(rng)
    state = create_train_state(
        model, init_rng, sample_batch,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
    )
    n_params = sum(p.size for p in jax.tree_util.tree_leaves(state.params))
    print(f"Model params: {n_params:,}")

    # Training
    os.makedirs(args.ckpt_dir, exist_ok=True)
    best_val_mae = float('inf')
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        # Train
        train_losses, train_maes = [], []
        for pt_batch in train_loader:
            batch = convert_batch(pt_batch)
            state, loss, mae = train_step(model, state, batch, normalizer)
            train_losses.append(float(loss))
            train_maes.append(float(mae))

        train_loss = np.mean(train_losses)
        train_mae = np.mean(train_maes)

        # Validation
        val_losses, val_maes = [], []
        for pt_batch in val_loader:
            batch = convert_batch(pt_batch)
            loss, mae, _ = eval_step(model, state, batch, normalizer)
            val_losses.append(float(loss))
            val_maes.append(float(mae))

        val_loss = np.mean(val_losses)
        val_mae = np.mean(val_maes)

        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch:3d} ({epoch_time:.1f}s) | "
              f"Train Loss: {train_loss:.4f} MAE: {train_mae:.4f} | "
              f"Val Loss: {val_loss:.4f} MAE: {val_mae:.4f}")

        # Checkpoint
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            patience_counter = 0
            save_checkpoint(
                os.path.join(args.ckpt_dir, 'best.pkl'),
                state, normalizer, epoch
            )
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break

    # Test
    print("\n" + "="*60)
    print("Testing")
    print("="*60)

    # Load best
    ckpt = load_checkpoint(os.path.join(args.ckpt_dir, 'best.pkl'))
    state = state.replace(params=ckpt['params'], batch_stats=ckpt.get('batch_stats'))
    normalizer.load_state_dict(ckpt['normalizer'])

    test_maes = []
    for pt_batch in test_loader:
        batch = convert_batch(pt_batch)
        loss, mae, _ = eval_step(model, state, batch, normalizer)
        test_maes.append(float(mae))

    test_mae = np.mean(test_maes)
    print(f"Test MAE: {test_mae:.4f}")

    return state, normalizer


# =============================================================================
# Prediction
# =============================================================================

def predict(args):
    """Load model and run inference."""
    print("\n" + "="*60)
    print("JAX Prediction")
    print("="*60)

    # Load checkpoint
    ckpt = load_checkpoint(args.ckpt)
    normalizer = Normalizer()
    normalizer.load_state_dict(ckpt['normalizer'])

    # Dataset (PyTorch)
    dataset = CIFData(root_dir=args.data_dir)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_pool,
    )

    # Sample batch for model init
    sample_pt_batch = next(iter(loader))
    sample_batch = convert_batch(sample_pt_batch)

    # Model
    model = CrystalGraphConvNet(
        orig_atom_fea_len=sample_batch['atom_fea'].shape[-1],
        nbr_fea_len=sample_batch['nbr_fea'].shape[-1],
        atom_fea_len=args.atom_fea_len,
        n_conv=args.n_conv,
        h_fea_len=args.h_fea_len,
        n_h=args.n_h,
        graphormer_layers=args.graphormer_layers,
        num_heads=args.num_heads,
    )

    # Create state with loaded params
    rng = random.PRNGKey(0)
    state = create_train_state(model, rng, sample_batch, learning_rate=1e-3)
    state = state.replace(params=ckpt['params'], batch_stats=ckpt.get('batch_stats'))

    # Predict
    all_preds, all_targets, all_ids = [], [], []
    for pt_batch in loader:
        batch = convert_batch(pt_batch)
        _, _, preds = eval_step(model, state, batch, normalizer)
        all_preds.extend(np.array(preds).tolist())
        all_targets.extend(np.array(batch['target'].squeeze(-1)).tolist())
        all_ids.extend(batch['cif_ids'])

    # Results
    print(f"\nPredictions: {len(all_preds)}")
    print(f"{'ID':20s} {'Target':>10s} {'Pred':>10s} {'Error':>10s}")
    print("-"*52)
    for i in range(min(10, len(all_preds))):
        err = abs(all_targets[i] - all_preds[i])
        print(f"{all_ids[i]:20s} {all_targets[i]:10.4f} {all_preds[i]:10.4f} {err:10.4f}")

    mae = np.mean(np.abs(np.array(all_targets) - np.array(all_preds)))
    print(f"\nOverall MAE: {mae:.4f}")


# =============================================================================
# Swap Benchmark
# =============================================================================

def swap_benchmark(args):
    """Benchmark swap operations."""
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
    print(f"Batch: {args.swap_batch}, N_swaps: {args.n_swaps}, Atoms: {N}")

    key = random.PRNGKey(42)
    key, subkey = random.split(key)

    structures = generate_structures(
        subkey, args.swap_batch,
        composition["Sr"], composition["Ti"], composition["Fe"],
        composition["O"], composition["VO"]
    )
    structures.block_until_ready()

    key, subkey = random.split(key)
    scores = random.normal(subkey, structures.shape)

    def bench(name, fn, warmup=3, trials=5):
        for _ in range(warmup):
            out = fn()
            if hasattr(out, 'block_until_ready'):
                out.block_until_ready()
            elif isinstance(out, tuple):
                out[0].block_until_ready()

        times = []
        for _ in range(trials):
            start = time.perf_counter()
            out = fn()
            if hasattr(out, 'block_until_ready'):
                out.block_until_ready()
            elif isinstance(out, tuple):
                out[0].block_until_ready()
            times.append(time.perf_counter() - start)

        print(f"{name:40s}: {np.mean(times)*1000:8.2f} ms")

    print()
    key, subkey = random.split(key)
    bench("apply_n_swaps (B-site)",
          lambda: apply_n_swaps(subkey, structures, scores, b_site_idx, 1, 2, args.n_swaps))

    key, subkey = random.split(key)
    bench("apply_n_swaps_both",
          lambda: apply_n_swaps_both(
              subkey, structures, scores,
              b_site_idx, o_site_idx,
              ATOM_TYPES["Ti"], ATOM_TYPES["Fe"],
              ATOM_TYPES["O"], ATOM_TYPES["VO"],
              args.n_swaps))

    bench("beam_search (k=4)",
          lambda: sample_sublattice_swap_beam(structures, b_site_idx, 1, 2, scores, 4))


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'predict', 'swap_benchmark'])
    parser.add_argument('--data_dir', type=str, default='./STFO_data')
    parser.add_argument('--ckpt_dir', type=str, default='./checkpoints_jax')
    parser.add_argument('--ckpt', type=str, default=None)

    # Training
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=0)

    # Model
    parser.add_argument('--atom_fea_len', type=int, default=64)
    parser.add_argument('--n_conv', type=int, default=3)
    parser.add_argument('--h_fea_len', type=int, default=128)
    parser.add_argument('--n_h', type=int, default=1)
    parser.add_argument('--graphormer_layers', type=int, default=1)
    parser.add_argument('--num_heads', type=int, default=4)

    # Swap benchmark
    parser.add_argument('--swap_batch', type=int, default=10000)
    parser.add_argument('--n_swaps', type=int, default=10000)

    args = parser.parse_args()

    if args.mode == 'train':
        train(args)
    elif args.mode == 'predict':
        if args.ckpt is None:
            raise ValueError("--ckpt required")
        predict(args)
    elif args.mode == 'swap_benchmark':
        swap_benchmark(args)


if __name__ == '__main__':
    main()
