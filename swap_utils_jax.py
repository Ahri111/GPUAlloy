"""
Sublattice-Constrained Swap Operations (JAX)
JAX implementation for TPU/GPU acceleration.
Fully vectorized, JIT-compatible.
For ABO3 perovskite structures:
- B-site: Ti ↔ Fe only
- O-site: O ↔ VO only
"""

import jax
import jax.numpy as jnp
from jax import random, lax
from functools import partial
from typing import Tuple, Optional, NamedTuple


# =============================================================================
# Data Structures
# =============================================================================

class SwapResult(NamedTuple):
    """Result of a swap operation."""
    swapped: jnp.ndarray      # [batch, N]
    indices: jnp.ndarray      # [batch, 2]


class BeamResult(NamedTuple):
    """Result of beam search."""
    swapped: jnp.ndarray      # [batch, beam, N]
    indices: jnp.ndarray      # [batch, beam, 2]
    log_probs: jnp.ndarray    # [batch, beam]


# =============================================================================
# Basic Swap Operations
# =============================================================================

@jax.jit
def swap_by_idx(x: jnp.ndarray, idx: jnp.ndarray) -> jnp.ndarray:
    """
    Swap elements at specified indices.
    Args:
        x: [batch, N] tensor
        idx: [batch, 2] indices to swap
    Returns:
        x_swapped: [batch, N]
    """
    batch_size = x.shape[0]
    batch_idx = jnp.arange(batch_size)

    # Get values at swap positions
    idx_a, idx_b = idx[:, 0], idx[:, 1]
    val_a = x[batch_idx, idx_a]
    val_b = x[batch_idx, idx_b]

    # Swap
    x_swapped = x.at[batch_idx, idx_a].set(val_b)
    x_swapped = x_swapped.at[batch_idx, idx_b].set(val_a)

    return x_swapped


@partial(jax.jit, static_argnums=(2, 3))
def sample_sublattice_swap(
    key: random.PRNGKey,
    atom_types: jnp.ndarray,
    sublattice_indices: Tuple[int, ...],  # static - tuple of indices
    type_a: int,
    type_b: int,
    scores: Optional[jnp.ndarray] = None,
) -> SwapResult:
    """
    Sample one constrained swap per batch element.
    Args:
        key: JAX random key
        atom_types: [batch, N] atom type indices
        sublattice_indices: tuple of indices in sublattice (static for JIT)
        type_a, type_b: types to swap
        scores: [batch, N] swap scores (None = uniform)
    Returns:
        SwapResult with swapped tensor and indices
    """
    sub_idx = jnp.array(sublattice_indices)
    batch_size, N = atom_types.shape
    M = len(sub_idx)

    # Extract sublattice
    sub_types = atom_types[:, sub_idx]  # [batch, M]

    # Type masks
    is_a = (sub_types == type_a)  # [batch, M]
    is_b = (sub_types == type_b)  # [batch, M]

    # Scores
    if scores is None:
        sub_scores = jnp.zeros((batch_size, M))
    else:
        sub_scores = scores[:, sub_idx]

    # Gumbel noise
    key_a, key_b = random.split(key)
    gumbel_a = random.gumbel(key_a, (batch_size, M))
    gumbel_b = random.gumbel(key_b, (batch_size, M))

    # Masked scores
    score_a = jnp.where(is_a, sub_scores + gumbel_a, -jnp.inf)
    score_b = jnp.where(is_b, sub_scores + gumbel_b, -jnp.inf)

    # Argmax selection
    local_a = jnp.argmax(score_a, axis=-1)  # [batch]
    local_b = jnp.argmax(score_b, axis=-1)  # [batch]

    # Map to global indices
    global_a = sub_idx[local_a]
    global_b = sub_idx[local_b]

    indices = jnp.stack([global_a, global_b], axis=-1)
    swapped = swap_by_idx(atom_types, indices)

    return SwapResult(swapped=swapped, indices=indices)


@partial(jax.jit, static_argnums=(2, 3))
def sample_sublattice_swap_deterministic(
    atom_types: jnp.ndarray,
    sublattice_indices: Tuple[int, ...],
    type_a: int,
    type_b: int,
    scores: jnp.ndarray,
) -> SwapResult:
    """
    Deterministic swap (no noise, pick highest scores).
    """
    sub_idx = jnp.array(sublattice_indices)
    batch_size = atom_types.shape[0]
    M = len(sub_idx)

    sub_types = atom_types[:, sub_idx]
    sub_scores = scores[:, sub_idx]

    is_a = (sub_types == type_a)
    is_b = (sub_types == type_b)

    score_a = jnp.where(is_a, sub_scores, -jnp.inf)
    score_b = jnp.where(is_b, sub_scores, -jnp.inf)

    local_a = jnp.argmax(score_a, axis=-1)
    local_b = jnp.argmax(score_b, axis=-1)

    global_a = sub_idx[local_a]
    global_b = sub_idx[local_b]

    indices = jnp.stack([global_a, global_b], axis=-1)
    swapped = swap_by_idx(atom_types, indices)

    return SwapResult(swapped=swapped, indices=indices)


# =============================================================================
# Beam Search
# =============================================================================

@partial(jax.jit, static_argnums=(2, 3, 5))
def sample_sublattice_swap_beam(
    atom_types: jnp.ndarray,
    sublattice_indices: Tuple[int, ...],
    type_a: int,
    type_b: int,
    scores: jnp.ndarray,
    beam_size: int = 4,
) -> BeamResult:
    """
    Beam search for top-k constrained swap candidates.
    Args:
        atom_types: [batch, N]
        sublattice_indices: tuple of sublattice indices
        type_a, type_b: types to swap
        scores: [batch, N] swap scores
        beam_size: number of candidates
    Returns:
        BeamResult with candidates, indices, and log_probs
    """
    sub_idx = jnp.array(sublattice_indices)
    batch_size, N = atom_types.shape
    M = len(sub_idx)

    sub_types = atom_types[:, sub_idx]
    sub_scores = scores[:, sub_idx]

    is_a = (sub_types == type_a)
    is_b = (sub_types == type_b)

    # Masked scores
    score_a = jnp.where(is_a, sub_scores, -jnp.inf)
    score_b = jnp.where(is_b, sub_scores, -jnp.inf)

    # Top-k from each type
    top_scores_a, top_local_a = lax.top_k(score_a, beam_size)  # [batch, beam]
    top_scores_b, top_local_b = lax.top_k(score_b, beam_size)  # [batch, beam]

    # All pairwise scores: [batch, beam, beam]
    pair_scores = top_scores_a[:, :, None] + top_scores_b[:, None, :]

    # Flatten and top-k
    pair_scores_flat = pair_scores.reshape(batch_size, -1)  # [batch, beam²]
    top_pair_scores, top_pair_idx = lax.top_k(pair_scores_flat, beam_size)

    # Convert flat index to (i, j)
    idx_a_beam = top_pair_idx // beam_size
    idx_b_beam = top_pair_idx % beam_size

    # Gather local indices
    batch_idx = jnp.arange(batch_size)[:, None]
    local_a_beam = top_local_a[batch_idx, idx_a_beam]  # [batch, beam]
    local_b_beam = top_local_b[batch_idx, idx_b_beam]  # [batch, beam]

    # Global indices
    global_a_beam = sub_idx[local_a_beam]
    global_b_beam = sub_idx[local_b_beam]

    indices_candidates = jnp.stack([global_a_beam, global_b_beam], axis=-1)

    # Generate swapped configurations using vmap
    def swap_single_beam(x, idx):
        """Swap for single batch, single beam."""
        return swap_by_idx(x[None, :], idx[None, :])[0]

    def swap_all_beams(x, indices):
        """Swap for single batch, all beams."""
        return jax.vmap(lambda idx: swap_single_beam(x, idx))(indices)

    swapped_candidates = jax.vmap(swap_all_beams)(atom_types, indices_candidates)

    # Log probabilities
    log_prob_a = jax.nn.log_softmax(score_a, axis=-1)
    log_prob_b = jax.nn.log_softmax(score_b, axis=-1)

    lp_a = log_prob_a[batch_idx, local_a_beam]
    lp_b = log_prob_b[batch_idx, local_b_beam]
    log_probs = lp_a + lp_b

    return BeamResult(
        swapped=swapped_candidates,
        indices=indices_candidates,
        log_probs=log_probs
    )


# =============================================================================
# Log Probability
# =============================================================================

@partial(jax.jit, static_argnums=(1, 2, 3))
def log_prob_sublattice_swap(
    scores: jnp.ndarray,
    sublattice_indices: Tuple[int, ...],
    type_a: int,
    type_b: int,
    atom_types: jnp.ndarray,
    swap_indices: jnp.ndarray,
) -> jnp.ndarray:
    """
    Compute log P(swap_indices | scores).
    Args:
        scores: [batch, N]
        sublattice_indices: tuple
        type_a, type_b: swapped types
        atom_types: [batch, N] BEFORE swap
        swap_indices: [batch, 2]
    Returns:
        log_probs: [batch]
    """
    sub_idx = jnp.array(sublattice_indices)
    batch_size = scores.shape[0]

    sub_types = atom_types[:, sub_idx]
    sub_scores = scores[:, sub_idx]

    is_a = (sub_types == type_a)
    is_b = (sub_types == type_b)

    score_a = jnp.where(is_a, sub_scores, -jnp.inf)
    score_b = jnp.where(is_b, sub_scores, -jnp.inf)

    log_prob_a = jax.nn.log_softmax(score_a, axis=-1)
    log_prob_b = jax.nn.log_softmax(score_b, axis=-1)

    # Find local indices
    global_a = swap_indices[:, 0]
    global_b = swap_indices[:, 1]

    # Map global to local
    local_a = jnp.argmax(sub_idx[None, :] == global_a[:, None], axis=-1)
    local_b = jnp.argmax(sub_idx[None, :] == global_b[:, None], axis=-1)

    batch_idx = jnp.arange(batch_size)
    lp_a = log_prob_a[batch_idx, local_a]
    lp_b = log_prob_b[batch_idx, local_b]

    return lp_a + lp_b


# =============================================================================
# Multiple Swap Steps (scan-based)
# =============================================================================

@partial(jax.jit, static_argnums=(3, 4, 5, 6))
def apply_n_swaps(
    key: random.PRNGKey,
    atom_types: jnp.ndarray,
    scores: Optional[jnp.ndarray],
    sublattice_indices: Tuple[int, ...],
    type_a: int,
    type_b: int,
    n_swaps: int,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Apply n swap steps using lax.scan for efficiency.
    Args:
        key: random key
        atom_types: [batch, N]
        scores: [batch, N] or None
        sublattice_indices: tuple
        type_a, type_b: types
        n_swaps: number of steps
    Returns:
        final: [batch, N]
        all_indices: [n_swaps, batch, 2]
    """
    keys = random.split(key, n_swaps)

    def scan_fn(carry, key_i):
        x = carry
        result = sample_sublattice_swap(
            key_i, x, sublattice_indices, type_a, type_b, scores
        )
        return result.swapped, result.indices

    final, all_indices = lax.scan(scan_fn, atom_types, keys)

    return final, all_indices


# =============================================================================
# Utility: Create sublattice mask
# =============================================================================

def get_sublattice_indices(
    atom_types: jnp.ndarray,
    target_types: Tuple[int, ...],
) -> Tuple[int, ...]:
    """
    Get indices where atom_types is in target_types.
    Returns as tuple for JIT compatibility.
    Args:
        atom_types: [N] single structure
        target_types: tuple of type indices
    Returns:
        tuple of indices
    """
    mask = jnp.zeros(atom_types.shape, dtype=bool)
    for t in target_types:
        mask = mask | (atom_types == t)
    indices = jnp.where(mask)[0]
    return tuple(indices.tolist())


# =============================================================================
# Example Usage
# =============================================================================

def example():
    """Example demonstrating JAX sublattice swap."""
    key = random.PRNGKey(42)

    # Create dummy structure: 32 atoms, types 0-4
    # Sr=0 (10), Ti=1 (8), Fe=2 (8), O=3 (5), VO=4 (1)
    atom_types = jnp.array([
        [0]*10 + [1]*8 + [2]*8 + [3]*5 + [4]*1  # batch=1
    ] * 4)  # batch=4

    # B-site indices (where Ti or Fe)
    b_site_indices = tuple(range(10, 26))  # indices 10-25

    # Scores (random for demo)
    key, subkey = random.split(key)
    scores = random.normal(subkey, atom_types.shape)

    print("=== JAX Sublattice Swap Demo ===")
    print(f"Input shape: {atom_types.shape}")
    print(f"B-site indices: {b_site_indices[:5]}... (total {len(b_site_indices)})")

    # Single swap
    key, subkey = random.split(key)
    result = sample_sublattice_swap(
        subkey, atom_types, b_site_indices, 1, 2, scores
    )
    print(f"\nSingle swap indices: {result.indices}")

    # Beam search
    beam_result = sample_sublattice_swap_beam(
        atom_types, b_site_indices, 1, 2, scores, beam_size=4
    )
    print(f"\nBeam search (k=4):")
    print(f"  Candidates shape: {beam_result.swapped.shape}")
    print(f"  Log probs: {beam_result.log_probs[0]}")

    # Multiple swaps
    key, subkey = random.split(key)
    final, all_idx = apply_n_swaps(
        subkey, atom_types, scores, b_site_indices, 1, 2, n_swaps=10
    )
    print(f"\nAfter 10 swaps:")
    print(f"  Final shape: {final.shape}")
    print(f"  All indices shape: {all_idx.shape}")

    return result, beam_result


if __name__ == "__main__":
    example()