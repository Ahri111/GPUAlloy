"""
JAX Native Data Loading Options

1. tf.data - TensorFlow 데이터 파이프라인 (가장 안정적)
2. No-loader - 전체 메모리 로드 (작은 데이터셋, MCMC용)

Usage:
    # tf.data 방식
    from data_loading_jax import create_tfdata_loader
    train_ds = create_tfdata_loader(data_dir, batch_size, split='train')
    for batch in train_ds:
        ...

    # No-loader 방식 (전체 메모리)
    from data_loading_jax import load_all_data
    data = load_all_data(data_dir)
    # 직접 인덱싱으로 배치 생성
"""

import os
import csv
import json
import warnings
from typing import Dict, Any, List, Tuple, Optional
from functools import lru_cache

import numpy as np
import jax
import jax.numpy as jnp
from jax import random

from pymatgen.core.structure import Structure
from pymatgen.core import DummySpecies


# =============================================================================
# Preprocessing (공통)
# =============================================================================

class GaussianDistance:
    """Gaussian distance filter."""
    def __init__(self, dmin: float, dmax: float, step: float, var: float = None):
        self.filter = np.arange(dmin, dmax + step, step)
        self.var = var if var else step

    def expand(self, distances: np.ndarray) -> np.ndarray:
        return np.exp(-(distances[..., np.newaxis] - self.filter) ** 2 / self.var ** 2)


def load_atom_embeddings(path: str) -> Dict[int, np.ndarray]:
    """Load atom embeddings from JSON."""
    with open(path) as f:
        data = json.load(f)
    return {int(k): np.array(v, dtype=np.float32) for k, v in data.items()}


def get_specie_number(specie) -> int:
    """Get atomic number, handling DummySpecies."""
    if isinstance(specie, DummySpecies):
        return 0
    return specie.number


def process_crystal(
    cif_path: str,
    atom_embeddings: Dict[int, np.ndarray],
    gdf: GaussianDistance,
    max_num_nbr: int = 12,
    radius: float = 8.0,
) -> Dict[str, np.ndarray]:
    """Process single CIF file to graph features."""
    crystal = Structure.from_file(cif_path)

    # Atom features
    atom_fea = np.vstack([
        atom_embeddings[get_specie_number(crystal[i].specie)]
        for i in range(len(crystal))
    ]).astype(np.float32)

    # Neighbor information
    all_nbrs = crystal.get_all_neighbors(radius, include_index=True)
    all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs]

    nbr_fea_idx, nbr_fea = [], []
    for nbr in all_nbrs:
        if len(nbr) < max_num_nbr:
            nbr_fea_idx.append(
                [x[2] for x in nbr] + [0] * (max_num_nbr - len(nbr))
            )
            nbr_fea.append(
                [x[1] for x in nbr] + [radius + 1.] * (max_num_nbr - len(nbr))
            )
        else:
            nbr_fea_idx.append([x[2] for x in nbr[:max_num_nbr]])
            nbr_fea.append([x[1] for x in nbr[:max_num_nbr]])

    nbr_fea_idx = np.array(nbr_fea_idx, dtype=np.int32)
    nbr_fea = gdf.expand(np.array(nbr_fea, dtype=np.float32)).astype(np.float32)

    return {
        'atom_fea': atom_fea,
        'nbr_fea': nbr_fea,
        'nbr_fea_idx': nbr_fea_idx,
        'n_atoms': len(crystal),
    }


# =============================================================================
# Option 1: tf.data (TensorFlow Data Pipeline)
# =============================================================================

def create_tfdata_loader(
    data_dir: str,
    batch_size: int,
    split: str = 'train',
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    max_num_nbr: int = 12,
    radius: float = 8.0,
    shuffle: bool = True,
    seed: int = 42,
):
    """
    Create tf.data Dataset for JAX training.

    Requires: pip install tensorflow
    """
    try:
        import tensorflow as tf
    except ImportError:
        raise ImportError("tensorflow required: pip install tensorflow")

    # Load metadata
    id_prop_file = os.path.join(data_dir, 'id_prop.csv')
    with open(id_prop_file) as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        id_prop_data = list(reader)

    # Shuffle deterministically
    np.random.seed(seed)
    np.random.shuffle(id_prop_data)

    # Split
    n = len(id_prop_data)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    if split == 'train':
        data = id_prop_data[:train_end]
    elif split == 'val':
        data = id_prop_data[train_end:val_end]
    else:
        data = id_prop_data[val_end:]

    # Load preprocessing
    atom_embeddings = load_atom_embeddings(os.path.join(data_dir, 'atom_init.json'))
    gdf = GaussianDistance(dmin=0.0, dmax=radius, step=0.2)

    # Process all data (with caching)
    @lru_cache(maxsize=None)
    def get_processed(cif_id: str):
        cif_path = os.path.join(data_dir, f"{cif_id}.cif")
        return process_crystal(cif_path, atom_embeddings, gdf, max_num_nbr, radius)

    # Generator function
    def generator():
        indices = list(range(len(data)))
        if shuffle:
            np.random.shuffle(indices)

        for i in indices:
            cif_id, target = data[i]
            processed = get_processed(cif_id)
            yield {
                'atom_fea': processed['atom_fea'],
                'nbr_fea': processed['nbr_fea'],
                'nbr_fea_idx': processed['nbr_fea_idx'],
                'target': np.array([float(target)], dtype=np.float32),
                'n_atoms': processed['n_atoms'],
            }

    # Get shapes from first sample
    sample = next(generator())
    atom_fea_len = sample['atom_fea'].shape[-1]
    nbr_fea_len = sample['nbr_fea'].shape[-1]

    output_signature = {
        'atom_fea': tf.TensorSpec(shape=(None, atom_fea_len), dtype=tf.float32),
        'nbr_fea': tf.TensorSpec(shape=(None, max_num_nbr, nbr_fea_len), dtype=tf.float32),
        'nbr_fea_idx': tf.TensorSpec(shape=(None, max_num_nbr), dtype=tf.int32),
        'target': tf.TensorSpec(shape=(1,), dtype=tf.float32),
        'n_atoms': tf.TensorSpec(shape=(), dtype=tf.int32),
    }

    ds = tf.data.Dataset.from_generator(generator, output_signature=output_signature)

    # Custom collate for variable-size graphs
    def collate_fn(batch):
        # Concatenate all atoms
        atom_fea = tf.concat([b['atom_fea'] for b in batch], axis=0)
        nbr_fea = tf.concat([b['nbr_fea'] for b in batch], axis=0)

        # Adjust neighbor indices
        nbr_fea_idx_list = []
        base_idx = 0
        crystal_atom_idx = []
        for b in batch:
            n = b['n_atoms']
            nbr_fea_idx_list.append(b['nbr_fea_idx'] + base_idx)
            crystal_atom_idx.append(tf.range(base_idx, base_idx + n))
            base_idx += n
        nbr_fea_idx = tf.concat(nbr_fea_idx_list, axis=0)

        targets = tf.stack([b['target'] for b in batch])

        return {
            'atom_fea': atom_fea,
            'nbr_fea': nbr_fea,
            'nbr_fea_idx': nbr_fea_idx,
            'crystal_atom_idx': crystal_atom_idx,
            'target': targets,
        }

    # Note: tf.data batching with collate is complex for variable-size graphs
    # Return unbatched for now, let user handle batching
    return ds, len(data)


# =============================================================================
# Option 2: No-Loader (Full Memory Load)
# =============================================================================

class FullMemoryDataset:
    """
    Load entire dataset into memory.
    Good for:
    - Small datasets
    - MCMC / swap optimization (no need for DataLoader)
    - Fast iteration
    """

    def __init__(
        self,
        data_dir: str,
        max_num_nbr: int = 12,
        radius: float = 8.0,
        seed: int = 42,
    ):
        self.data_dir = data_dir
        self.max_num_nbr = max_num_nbr
        self.radius = radius

        # Load metadata
        id_prop_file = os.path.join(data_dir, 'id_prop.csv')
        with open(id_prop_file) as f:
            reader = csv.reader(f)
            next(reader)
            self.id_prop_data = list(reader)

        np.random.seed(seed)
        np.random.shuffle(self.id_prop_data)

        # Preprocessing tools
        self.atom_embeddings = load_atom_embeddings(
            os.path.join(data_dir, 'atom_init.json')
        )
        self.gdf = GaussianDistance(dmin=0.0, dmax=radius, step=0.2)

        # Load all data
        print(f"Loading {len(self.id_prop_data)} structures into memory...")
        self.data = []
        for i, (cif_id, target) in enumerate(self.id_prop_data):
            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{len(self.id_prop_data)}")

            cif_path = os.path.join(data_dir, f"{cif_id}.cif")
            processed = process_crystal(
                cif_path, self.atom_embeddings, self.gdf,
                max_num_nbr, radius
            )
            processed['target'] = np.array([float(target)], dtype=np.float32)
            processed['cif_id'] = cif_id
            self.data.append(processed)

        print(f"Loaded {len(self.data)} structures")

        # Precompute all targets for normalizer
        self.all_targets = np.array([d['target'][0] for d in self.data])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def get_batch(self, indices: List[int]) -> Dict[str, jnp.ndarray]:
        """Get a collated batch by indices."""
        batch_atom_fea = []
        batch_nbr_fea = []
        batch_nbr_fea_idx = []
        crystal_atom_idx = []
        batch_target = []
        base_idx = 0

        for idx in indices:
            d = self.data[idx]
            n = d['n_atoms']

            batch_atom_fea.append(d['atom_fea'])
            batch_nbr_fea.append(d['nbr_fea'])
            batch_nbr_fea_idx.append(d['nbr_fea_idx'] + base_idx)
            crystal_atom_idx.append(jnp.arange(n) + base_idx)
            batch_target.append(d['target'])

            base_idx += n

        return {
            'atom_fea': jnp.concatenate(batch_atom_fea, axis=0),
            'nbr_fea': jnp.concatenate(batch_nbr_fea, axis=0),
            'nbr_fea_idx': jnp.concatenate(batch_nbr_fea_idx, axis=0),
            'crystal_atom_idx': crystal_atom_idx,
            'target': jnp.stack(batch_target, axis=0),
        }

    def get_split_indices(
        self,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get train/val/test indices."""
        n = len(self.data)
        indices = np.arange(n)

        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)

        return (
            indices[:train_end],
            indices[train_end:val_end],
            indices[val_end:],
        )

    def iterate_batches(
        self,
        indices: np.ndarray,
        batch_size: int,
        shuffle: bool = True,
        rng: Optional[random.PRNGKey] = None,
    ):
        """Iterate over batches (generator)."""
        # Copy to avoid modifying original
        indices = np.array(indices, copy=True)

        if shuffle:
            if rng is not None:
                perm = random.permutation(rng, len(indices))
                indices = indices[np.array(perm)]  # Convert JAX array to numpy
            else:
                np.random.shuffle(indices)

        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start:start + batch_size]
            yield self.get_batch(batch_indices.tolist())


def load_all_data(
    data_dir: str,
    max_num_nbr: int = 12,
    radius: float = 8.0,
    seed: int = 42,
) -> FullMemoryDataset:
    """Convenience function to load full dataset."""
    return FullMemoryDataset(data_dir, max_num_nbr, radius, seed)


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./STFO_data')
    parser.add_argument('--mode', type=str, default='memory',
                        choices=['memory', 'tfdata'])
    args = parser.parse_args()

    if args.mode == 'memory':
        print("=== Full Memory Load ===")
        dataset = load_all_data(args.data_dir)
        print(f"Dataset size: {len(dataset)}")
        print(f"Target mean: {dataset.all_targets.mean():.4f}")
        print(f"Target std: {dataset.all_targets.std():.4f}")

        train_idx, val_idx, test_idx = dataset.get_split_indices()
        print(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

        # Get one batch
        batch = dataset.get_batch([0, 1, 2])
        print(f"Batch atom_fea shape: {batch['atom_fea'].shape}")

    elif args.mode == 'tfdata':
        print("=== tf.data ===")
        ds, n = create_tfdata_loader(args.data_dir, batch_size=16)
        print(f"Dataset size: {n}")
