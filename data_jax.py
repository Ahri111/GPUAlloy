"""
JAX-compatible data loading for crystal graph neural networks.
This module provides JAX/NumPy versions of the data loading utilities.
CIF parsing still uses pymatgen (Python), but tensors are JAX arrays.
Note: pymatgen is required for CIF file parsing and cannot be replaced with JAX.
"""

from __future__ import annotations

import csv
import functools
import json
import os
import random
import warnings
from typing import List, Tuple, Optional, Dict, Any, Iterator

import numpy as np
import jax
import jax.numpy as jnp
from pymatgen.core.structure import Structure
from pymatgen.core import DummySpecies


class GaussianDistance:
    """
    Expands the distance by Gaussian basis.
    Unit: angstrom
    """

    def __init__(self, dmin: float, dmax: float, step: float, var: Optional[float] = None):
        """
        Parameters
        ----------
        dmin: float
            Minimum interatomic distance
        dmax: float
            Maximum interatomic distance
        step: float
            Step size for the Gaussian filter
        var: float, optional
            Variance of Gaussian (default: step)
        """
        assert dmin < dmax
        assert dmax - dmin > step
        self.filter = np.arange(dmin, dmax + step, step)
        self.var = var if var is not None else step

    def expand(self, distances: np.ndarray) -> np.ndarray:
        """
        Apply Gaussian distance filter to a numpy distance array.
        Parameters
        ----------
        distances: np.ndarray
            A distance matrix of any shape
        Returns
        -------
        expanded_distance: np.ndarray
            Expanded distance matrix with the last dimension of length len(self.filter)
        """
        return np.exp(-(distances[..., np.newaxis] - self.filter) ** 2 / self.var ** 2)


class AtomInitializer:
    """
    Base class for initializing the vector representation for atoms.
    """

    def __init__(self, atom_types: set):
        self.atom_types = set(atom_types)
        self._embedding: Dict[int, np.ndarray] = {}

    def get_atom_fea(self, atom_type: int) -> np.ndarray:
        assert atom_type in self.atom_types
        return self._embedding[atom_type]

    def load_state_dict(self, state_dict: Dict[int, np.ndarray]) -> None:
        self._embedding = state_dict
        self.atom_types = set(self._embedding.keys())
        self._decodedict = {idx: atom_type for atom_type, idx in self._embedding.items()}

    def state_dict(self) -> Dict[int, np.ndarray]:
        return self._embedding

    def decode(self, idx: int) -> int:
        if not hasattr(self, '_decodedict'):
            self._decodedict = {idx: atom_type for atom_type, idx in self._embedding.items()}
        return self._decodedict[idx]


class AtomCustomJSONInitializer(AtomInitializer):
    """
    Initialize atom feature vectors using a JSON file.
    Parameters
    ----------
    elem_embedding_file: str
        The path to the .json file
    """

    def __init__(self, elem_embedding_file: str):
        with open(elem_embedding_file) as f:
            elem_embedding = json.load(f)
        elem_embedding = {int(key): value for key, value in elem_embedding.items()}
        atom_types = set(elem_embedding.keys())
        super().__init__(atom_types)
        for key, value in elem_embedding.items():
            self._embedding[key] = np.array(value, dtype=np.float32)


class CrystalGraphData:
    """
    Container for a single crystal graph in JAX format.
    """

    def __init__(
        self,
        atom_fea: jnp.ndarray,
        nbr_fea: jnp.ndarray,
        nbr_fea_idx: jnp.ndarray,
        target: jnp.ndarray,
        cif_id: str
    ):
        self.atom_fea = atom_fea          # (n_atoms, atom_fea_len)
        self.nbr_fea = nbr_fea            # (n_atoms, max_num_nbr, nbr_fea_len)
        self.nbr_fea_idx = nbr_fea_idx    # (n_atoms, max_num_nbr)
        self.target = target              # (1,)
        self.cif_id = cif_id

    def to_dict(self) -> Dict[str, Any]:
        return {
            'atom_fea': self.atom_fea,
            'nbr_fea': self.nbr_fea,
            'nbr_fea_idx': self.nbr_fea_idx,
            'target': self.target,
            'cif_id': self.cif_id
        }


class CIFDataJAX:
    """
    JAX-compatible CIF dataset loader.
    The dataset should have the following directory structure:
    root_dir
    ├── id_prop.csv
    ├── atom_init.json
    ├── id0.cif
    ├── id1.cif
    ├── ...
    Parameters
    ----------
    root_dir: str
        The path to the root directory of the dataset
    max_num_nbr: int
        The maximum number of neighbors while constructing the crystal graph
    radius: float
        The cutoff radius for searching neighbors
    dmin: float
        The minimum distance for constructing GaussianDistance
    step: float
        The step size for constructing GaussianDistance
    random_seed: int
        Random seed for shuffling the dataset
    """

    def __init__(
        self,
        root_dir: str,
        max_num_nbr: int = 12,
        radius: float = 8.0,
        dmin: float = 0.0,
        step: float = 0.2,
        random_seed: int = 123
    ):
        self.root_dir = root_dir
        self.max_num_nbr = max_num_nbr
        self.radius = radius

        assert os.path.exists(root_dir), 'root_dir does not exist!'

        id_prop_file = os.path.join(self.root_dir, 'id_prop.csv')
        assert os.path.exists(id_prop_file), 'id_prop.csv does not exist!'

        with open(id_prop_file) as f:
            reader = csv.reader(f)
            next(reader)  # Skip header row (matching PyTorch version)
            self.id_prop_data = [row for row in reader]

        random.seed(random_seed)
        random.shuffle(self.id_prop_data)

        atom_init_file = os.path.join(self.root_dir, 'atom_init.json')
        assert os.path.exists(atom_init_file), 'atom_init.json does not exist!'

        self.ari = AtomCustomJSONInitializer(atom_init_file)
        self.gdf = GaussianDistance(dmin=dmin, dmax=self.radius, step=step)

        # Cache for loaded structures
        self._cache: Dict[int, CrystalGraphData] = {}

    def __len__(self) -> int:
        return len(self.id_prop_data)

    @staticmethod
    def get_specie_number(specie):
        """Get atomic number, handling DummySpecies (e.g., vacancies)."""
        if isinstance(specie, DummySpecies):
            return 0  # Return 0 for dummy species like vacancies (VO)
        return specie.number

    def __getitem__(self, idx: int) -> CrystalGraphData:
        if idx in self._cache:
            return self._cache[idx]

        cif_id, target = self.id_prop_data[idx]
        crystal = Structure.from_file(os.path.join(self.root_dir, cif_id + '.cif'))

        # Atom features (with DummySpecies handling)
        atom_fea = np.vstack([
            self.ari.get_atom_fea(self.get_specie_number(crystal[i].specie))
            for i in range(len(crystal))
        ]).astype(np.float32)

        # Neighbor information
        all_nbrs = crystal.get_all_neighbors(self.radius, include_index=True)
        all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs]

        nbr_fea_idx, nbr_fea = [], []
        for nbr in all_nbrs:
            if len(nbr) < self.max_num_nbr:
                warnings.warn(
                    f'{cif_id} not find enough neighbors to build graph. '
                    'If it happens frequently, consider increase radius.'
                )
                nbr_fea_idx.append(
                    list(map(lambda x: x[2], nbr)) +
                    [0] * (self.max_num_nbr - len(nbr))
                )
                nbr_fea.append(
                    list(map(lambda x: x[1], nbr)) +
                    [self.radius + 1.] * (self.max_num_nbr - len(nbr))
                )
            else:
                nbr_fea_idx.append(list(map(lambda x: x[2], nbr[:self.max_num_nbr])))
                nbr_fea.append(list(map(lambda x: x[1], nbr[:self.max_num_nbr])))

        nbr_fea_idx = np.array(nbr_fea_idx, dtype=np.int32)
        nbr_fea = np.array(nbr_fea, dtype=np.float32)
        nbr_fea = self.gdf.expand(nbr_fea).astype(np.float32)

        # Convert to JAX arrays
        data = CrystalGraphData(
            atom_fea=jnp.array(atom_fea),
            nbr_fea=jnp.array(nbr_fea),
            nbr_fea_idx=jnp.array(nbr_fea_idx),
            target=jnp.array([float(target)], dtype=jnp.float32),
            cif_id=cif_id
        )

        self._cache[idx] = data
        return data

    def get_batch(self, indices: List[int]) -> Dict[str, Any]:
        """
        Get a batched representation of multiple crystals.
        Returns a dictionary with concatenated features and indices for pooling.
        """
        batch_atom_fea = []
        batch_nbr_fea = []
        batch_nbr_fea_idx = []
        crystal_atom_idx = []
        batch_target = []
        batch_cif_ids = []
        base_idx = 0

        for idx in indices:
            data = self[idx]
            n_i = data.atom_fea.shape[0]

            batch_atom_fea.append(data.atom_fea)
            batch_nbr_fea.append(data.nbr_fea)
            batch_nbr_fea_idx.append(data.nbr_fea_idx + base_idx)
            crystal_atom_idx.append(jnp.arange(n_i) + base_idx)
            batch_target.append(data.target)
            batch_cif_ids.append(data.cif_id)

            base_idx += n_i

        return {
            'atom_fea': jnp.concatenate(batch_atom_fea, axis=0),
            'nbr_fea': jnp.concatenate(batch_nbr_fea, axis=0),
            'nbr_fea_idx': jnp.concatenate(batch_nbr_fea_idx, axis=0),
            'crystal_atom_idx': crystal_atom_idx,
            'target': jnp.stack(batch_target, axis=0),
            'cif_ids': batch_cif_ids
        }


class JAXDataLoader:
    """
    Simple data loader for JAX that yields batches.
    Unlike PyTorch DataLoader, this yields JAX arrays directly.
    """

    def __init__(
        self,
        dataset: CIFDataJAX,
        batch_size: int = 64,
        shuffle: bool = True,
        drop_last: bool = False,
        rng_key: Optional[jax.random.PRNGKey] = None
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.rng_key = rng_key if rng_key is not None else jax.random.PRNGKey(0)

    def __len__(self) -> int:
        n = len(self.dataset)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        indices = list(range(len(self.dataset)))

        if self.shuffle:
            self.rng_key, subkey = jax.random.split(self.rng_key)
            perm = jax.random.permutation(subkey, len(indices))
            indices = [indices[i] for i in np.array(perm)]

        for start in range(0, len(indices), self.batch_size):
            end = start + self.batch_size
            batch_indices = indices[start:end]

            if self.drop_last and len(batch_indices) < self.batch_size:
                break

            yield self.dataset.get_batch(batch_indices)


def get_train_val_test_loader_jax(
    dataset: CIFDataJAX,
    batch_size: int = 64,
    train_ratio: Optional[float] = None,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    return_test: bool = False,
    train_size: Optional[int] = None,
    val_size: Optional[int] = None,
    test_size: Optional[int] = None,
    rng_key: Optional[jax.random.PRNGKey] = None
) -> Tuple[JAXDataLoader, ...]:
    """
    Create train/val/test data loaders for JAX.
    Parameters
    ----------
    dataset: CIFDataJAX
        The dataset to split
    batch_size: int
        Batch size for all loaders
    train_ratio, val_ratio, test_ratio: float
        Ratios for splitting data
    return_test: bool
        Whether to return test loader
    train_size, val_size, test_size: int, optional
        Explicit sizes (override ratios)
    rng_key: jax.random.PRNGKey
        Random key for shuffling
    Returns
    -------
    Tuple of JAXDataLoader instances
    """
    total_size = len(dataset)

    if train_size is None:
        if train_ratio is None:
            assert val_ratio + test_ratio < 1
            train_ratio = 1 - val_ratio - test_ratio
        train_size = int(train_ratio * total_size)

    if test_size is None:
        test_size = int(test_ratio * total_size)

    if val_size is None:
        val_size = int(val_ratio * total_size)

    indices = list(range(total_size))

    train_indices = indices[:train_size]
    if test_size > 0:
        val_indices = indices[-(val_size + test_size):-test_size]
        test_indices = indices[-test_size:]
    else:
        val_indices = indices[-val_size:]
        test_indices = []

    if rng_key is None:
        rng_key = jax.random.PRNGKey(42)

    # Create subset datasets
    train_loader = SubsetJAXDataLoader(dataset, train_indices, batch_size, shuffle=True, rng_key=rng_key)

    rng_key, subkey = jax.random.split(rng_key)
    val_loader = SubsetJAXDataLoader(dataset, val_indices, batch_size, shuffle=False, rng_key=subkey)

    if return_test:
        rng_key, subkey = jax.random.split(rng_key)
        test_loader = SubsetJAXDataLoader(dataset, test_indices, batch_size, shuffle=False, rng_key=subkey)
        return train_loader, val_loader, test_loader

    return train_loader, val_loader


class SubsetJAXDataLoader:
    """DataLoader for a subset of indices."""

    def __init__(
        self,
        dataset: CIFDataJAX,
        indices: List[int],
        batch_size: int = 64,
        shuffle: bool = True,
        rng_key: Optional[jax.random.PRNGKey] = None
    ):
        self.dataset = dataset
        self.indices = indices
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rng_key = rng_key if rng_key is not None else jax.random.PRNGKey(0)

    def __len__(self) -> int:
        return (len(self.indices) + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        indices = list(self.indices)

        if self.shuffle:
            self.rng_key, subkey = jax.random.split(self.rng_key)
            perm = jax.random.permutation(subkey, len(indices))
            indices = [indices[int(i)] for i in np.array(perm)]

        for start in range(0, len(indices), self.batch_size):
            end = min(start + self.batch_size, len(indices))
            batch_indices = indices[start:end]
            yield self.dataset.get_batch(batch_indices)


# Utility functions for converting between PyTorch and JAX formats

def torch_to_jax(tensor) -> jnp.ndarray:
    """Convert PyTorch tensor to JAX array."""
    import torch
    if isinstance(tensor, torch.Tensor):
        return jnp.array(tensor.detach().cpu().numpy())
    return jnp.array(tensor)


def jax_to_torch(array: jnp.ndarray):
    """Convert JAX array to PyTorch tensor."""
    import torch
    return torch.from_numpy(np.array(array))


def convert_batch_torch_to_jax(batch: tuple) -> Dict[str, Any]:
    """
    Convert a PyTorch collated batch to JAX format.
    Args:
        batch: Tuple from PyTorch collate_pool function
    Returns:
        Dictionary with JAX arrays
    """
    (atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx), target, cif_ids = batch

    return {
        'atom_fea': torch_to_jax(atom_fea),
        'nbr_fea': torch_to_jax(nbr_fea),
        'nbr_fea_idx': torch_to_jax(nbr_fea_idx),
        'crystal_atom_idx': [torch_to_jax(idx) for idx in crystal_atom_idx],
        'target': torch_to_jax(target),
        'cif_ids': cif_ids
    }