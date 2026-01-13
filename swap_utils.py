import torch
import torch.nn.functional as F
from typing import Tuple, Optional


def parse_poscar_string(poscar_str: str) -> dict:
    """Parse POSCAR string into structured dict with sublattice info"""
    lines = [l.strip() for l in poscar_str.strip().split('\n')]

    # Parse header
    cell_name = lines[0]
    latt_const = float(lines[1])

    # Lattice vectors
    base = []
    for i in range(2, 5):
        base.append([float(x) for x in lines[i].split()])

    # Element names and counts
    ele_names = lines[5].split()
    atom_nums = [int(x) for x in lines[6].split()]
    atom_sum = sum(atom_nums)

    # Coordinate type
    coord_type = lines[7]

    # Positions
    positions = []
    for i in range(8, 8 + atom_sum):
        parts = lines[i].split()
        pos = [float(parts[0]), float(parts[1]), float(parts[2])]
        positions.append(pos)

    return {
        'CellName': cell_name,
        'LattConst': latt_const,
        'Base': base,
        'EleName': ele_names,
        'AtomNum': atom_nums,
        'AtomSum': atom_sum,
        'LatType': coord_type,
        'LattPnt': positions
    }

# Before it goest to tensor, we handle all the sublattice masks here in CPU since GPU is not adequate for string processing
def poscar_to_tensors(poscar: dict, device='cpu') -> dict:

    """Convert POSCAR to tensors with sublattice masks"""
    # Positions [N, 3]
    positions = torch.tensor(poscar['LattPnt'], dtype=torch.float32, device=device)
    # Atom types [N]
    atom_types = []

    for type_idx, count in enumerate(poscar['AtomNum']):
        atom_types.extend([type_idx] * count)
    atom_types = torch.tensor(atom_types, dtype=torch.long, device=device)
    # Lattice [3, 3]
    lattice = torch.tensor(poscar['Base'], dtype=torch.float32, device=device)
    # Type mapping: Sr=0, Ti=1, Fe=2, O=3, VO=4
    ele_names = poscar['EleName']
    type_map = {name: idx for idx, name in enumerate(ele_names)}
    # Sublattice masks
    b_site_mask = (atom_types == type_map['Ti']) | (atom_types == type_map['Fe'])
    o_site_mask = (atom_types == type_map['O']) | (atom_types == type_map['VO'])

    return {
        'positions': positions,
        'atom_types': atom_types,
        'lattice': lattice,
        'b_site_mask': b_site_mask,
        'o_site_mask': o_site_mask,
        'element_names': ele_names,
        'type_map': type_map,
        'atom_counts': poscar['AtomNum']
    }

@torch.no_grad()
def swap_by_idx(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:

    """
    Swap elements at specified indices (fully parallel).
    Args:
        x: [batch, N] tensor
        idx: [batch, 2] indices to swap
    Returns:
        x_swapped: [batch, N]
    """
    first = x.gather(-1, idx[..., 0:1])
    second = x.gather(-1, idx[..., 1:2])

    x_swapped = x.clone()
    x_swapped.scatter_(-1, idx[..., 0:1], second)
    x_swapped.scatter_(-1, idx[..., 1:2], first)

    return x_swapped


@torch.no_grad()
def sample_sublattice_swap(
    atom_types: torch.Tensor,
    sublattice_mask: torch.Tensor,
    type_a: int,
    type_b: int,
    scores: Optional[torch.Tensor] = None,
    deterministic: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sample one constrained swap per batch element (Gumbel-Max).
    Args:
        atom_types: [batch, N] atom type indices
        sublattice_mask: [N] boolean mask for sublattice
        type_a, type_b: types to swap (must be different)
        scores: [batch, N] swap scores (None = uniform)
        deterministic: if True, pick highest score (no noise)
    Returns:
        swapped: [batch, N] after swap
        indices: [batch, 2] swapped positions
    """
    device = atom_types.device
    batch_size, N = atom_types.shape

    # Extract sublattice
    sub_idx = torch.where(sublattice_mask)[0]
    M = len(sub_idx)
    sub_types = atom_types[:, sub_idx]

    # Type masks
    is_a = (sub_types == type_a)
    is_b = (sub_types == type_b)

    # Scores
    if scores is None:
        sub_scores = torch.zeros(batch_size, M, device=device)
    else:
        sub_scores = scores[:, sub_idx]

    # Gumbel noise (skip if deterministic)
    if not deterministic:
        noise = torch.rand(batch_size, M, device=device).clamp(min=1e-10)
        gumbel = -torch.log(-torch.log(noise))
        sub_scores = sub_scores + gumbel

    # Masked scores
    score_a = sub_scores.masked_fill(~is_a, float('-inf'))
    score_b = sub_scores.masked_fill(~is_b, float('-inf'))

    # Select one from each type
    local_a = torch.argmax(score_a, dim=-1)
    local_b = torch.argmax(score_b, dim=-1)

    # Map to global indices
    global_a = sub_idx[local_a]
    global_b = sub_idx[local_b]

    indices = torch.stack([global_a, global_b], dim=-1)
    swapped = swap_by_idx(atom_types, indices)

    return swapped, indices

@torch.no_grad()
def apply_n_swaps(
    atom_types: torch.Tensor,       # [batch, N]
    b_site_mask: torch.Tensor,      # [N]
    o_site_mask: torch.Tensor,      # [N]
    type_map: dict,
    n_swaps: int,
    swap_mode: str = 'both'
) -> tuple:
    """
    Apply n swap steps (batch parallel at each step)

    Args:
        atom_types: [batch, N]
        b_site_mask, o_site_mask: sublattice masks
        type_map: element name -> type index
        n_swaps: number of swap steps
        swap_mode: 'B-site', 'O-site', or 'both'
    Returns:
        final: [batch, N] after all swaps
        history: list of (sublattice, indices) tuples
    """
    current = atom_types.clone()
    history = []

    ti, fe = type_map['Ti'], type_map['Fe']
    o, vo = type_map['O'], type_map['VO']

    for step in range(n_swaps):
        # Choose sublattice
        if swap_mode == 'B-site':
            do_b = True
        elif swap_mode == 'O-site':
            do_b = False
        else:
            do_b = torch.rand(1).item() < 0.5

        if do_b:
            current, idx = sample_sublattice_swap(current, b_site_mask, ti, fe)
            history.append(('B', idx.clone()))
        else:
            current, idx = sample_sublattice_swap(current, o_site_mask, o, vo)
            history.append(('O', idx.clone()))

    return current, history

@torch.no_grad()
def sample_sublattice_swap_beam(
    atom_types: torch.Tensor,
    sublattice_mask: torch.Tensor,
    type_a: int,
    type_b: int,
    scores: Optional[torch.Tensor] = None,
    beam_size: int = 4,
    return_log_probs: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    Beam search for top-k constrained swap candidates.
    Returns top beam_size (type_a, type_b) pairs ranked by score.
    Args:
        atom_types: [batch, N] atom type indices
        sublattice_mask: [N] boolean mask for sublattice
        type_a, type_b: types to swap
        scores: [batch, N] swap scores (None = uniform)
        beam_size: number of candidates to return
        return_log_probs: whether to compute log probabilities
    Returns:
        swapped_candidates: [batch, beam_size, N] all swap results
        indices_candidates: [batch, beam_size, 2] swap indices
        log_probs: [batch, beam_size] log probabilities (if requested)
    """
    device = atom_types.device
    batch_size, N = atom_types.shape

    # Extract sublattice
    sub_idx = torch.where(sublattice_mask)[0]
    M = len(sub_idx)
    sub_types = atom_types[:, sub_idx]

    # Type masks
    is_a = (sub_types == type_a)  # [batch, M]
    is_b = (sub_types == type_b)  # [batch, M]

    # Scores
    if scores is None:
        sub_scores = torch.zeros(batch_size, M, device=device)
    else:
        sub_scores = scores[:, sub_idx]

    # Get top-k from each type
    score_a = sub_scores.masked_fill(~is_a, float('-inf'))
    score_b = sub_scores.masked_fill(~is_b, float('-inf'))

    # Top-k indices for each type
    k_a = min(beam_size, is_a.sum(dim=-1).min().item())
    k_b = min(beam_size, is_b.sum(dim=-1).min().item())

    top_scores_a, top_local_a = torch.topk(score_a, k=k_a, dim=-1)  # [batch, k_a]
    top_scores_b, top_local_b = torch.topk(score_b, k=k_b, dim=-1)  # [batch, k_b]

    # Compute all pairwise scores: score[i,j] = score_a[i] + score_b[j]
    # Shape: [batch, k_a, k_b]
    pair_scores = top_scores_a.unsqueeze(-1) + top_scores_b.unsqueeze(-2)

    # Flatten and get top beam_size pairs
    pair_scores_flat = pair_scores.view(batch_size, -1)  # [batch, k_a * k_b]

    actual_beam = min(beam_size, k_a * k_b)
    top_pair_scores, top_pair_idx = torch.topk(
        pair_scores_flat, k=actual_beam, dim=-1
    )  # [batch, beam]

    # Convert flat index back to (i, j)
    idx_a_beam = top_pair_idx // k_b  # [batch, beam]
    idx_b_beam = top_pair_idx % k_b   # [batch, beam]

    # Gather actual local indices
    local_a_beam = torch.gather(top_local_a, -1, idx_a_beam)  # [batch, beam]
    local_b_beam = torch.gather(top_local_b, -1, idx_b_beam)  # [batch, beam]

    # Map to global indices
    global_a_beam = sub_idx[local_a_beam]  # [batch, beam]
    global_b_beam = sub_idx[local_b_beam]  # [batch, beam]

    # Stack indices: [batch, beam, 2]
    indices_candidates = torch.stack([global_a_beam, global_b_beam], dim=-1)

    # Generate all swapped configurations
    # [batch, beam, N]
    swapped_candidates = atom_types.unsqueeze(1).expand(-1, actual_beam, -1).clone()

    for b in range(actual_beam):
        idx_b_tensor = indices_candidates[:, b, :]  # [batch, 2]
        swapped_candidates[:, b, :] = swap_by_idx(atom_types, idx_b_tensor)

    # Compute log probabilities if requested
    log_probs = None
    if return_log_probs:
        log_probs = compute_swap_log_prob_beam(
            sub_scores, is_a, is_b, local_a_beam, local_b_beam
        )

    return swapped_candidates, indices_candidates, log_probs


def compute_swap_log_prob_beam(
    sub_scores: torch.Tensor,
    is_a: torch.Tensor,
    is_b: torch.Tensor,
    local_a: torch.Tensor,
    local_b: torch.Tensor,
) -> torch.Tensor:
    """
    Compute log probability of selecting (local_a, local_b) pairs.
    P(a, b) = P(a | type_a) * P(b | type_b)
            = softmax(scores_a)[a] * softmax(scores_b)[b]
    Args:
        sub_scores: [batch, M] scores on sublattice
        is_a, is_b: [batch, M] type masks
        local_a, local_b: [batch, beam] selected local indices
    Returns:
        log_probs: [batch, beam]
    """
    # Masked log-softmax for each type
    score_a = sub_scores.masked_fill(~is_a, float('-inf'))
    score_b = sub_scores.masked_fill(~is_b, float('-inf'))

    log_prob_a = F.log_softmax(score_a, dim=-1)  # [batch, M]
    log_prob_b = F.log_softmax(score_b, dim=-1)  # [batch, M]

    # Gather log probs at selected indices
    lp_a = torch.gather(log_prob_a, -1, local_a)  # [batch, beam]
    lp_b = torch.gather(log_prob_b, -1, local_b)  # [batch, beam]

    # Joint log prob (assuming independence)
    return lp_a + lp_b


def compute_swap_log_prob_beam(
    sub_scores: torch.Tensor,
    is_a: torch.Tensor,
    is_b: torch.Tensor,
    local_a: torch.Tensor,
    local_b: torch.Tensor,
) -> torch.Tensor:
    """
    Compute log probability of selecting (local_a, local_b) pairs.
    P(a, b) = P(a | type_a) * P(b | type_b)
            = softmax(scores_a)[a] * softmax(scores_b)[b]
    Args:
        sub_scores: [batch, M] scores on sublattice
        is_a, is_b: [batch, M] type masks
        local_a, local_b: [batch, beam] selected local indices
    Returns:
        log_probs: [batch, beam]
    """
    # Masked log-softmax for each type
    score_a = sub_scores.masked_fill(~is_a, float('-inf'))
    score_b = sub_scores.masked_fill(~is_b, float('-inf'))

    log_prob_a = F.log_softmax(score_a, dim=-1)  # [batch, M]
    log_prob_b = F.log_softmax(score_b, dim=-1)  # [batch, M]

    # Gather log probs at selected indices
    lp_a = torch.gather(log_prob_a, -1, local_a)  # [batch, beam]
    lp_b = torch.gather(log_prob_b, -1, local_b)  # [batch, beam]

    # Joint log prob (assuming independence)
    return lp_a + lp_b


def log_prob_sublattice_swap(
    scores: torch.Tensor,
    sublattice_mask: torch.Tensor,
    type_a: int,
    type_b: int,
    atom_types: torch.Tensor,
    swap_indices: torch.Tensor,
) -> torch.Tensor:
    """
    Compute log P(swap_indices | scores) for REINFORCE training.
    Args:
        scores: [batch, N] swap scores from neural network
        sublattice_mask: [N] boolean mask
        type_a, type_b: swapped types
        atom_types: [batch, N] atom types BEFORE swap
        swap_indices: [batch, 2] indices that were swapped
    Returns:
        log_probs: [batch]
    """
    device = scores.device
    batch_size = scores.shape[0]

    # Extract sublattice
    sub_idx = torch.where(sublattice_mask)[0]
    sub_types = atom_types[:, sub_idx]
    sub_scores = scores[:, sub_idx]

    # Type masks
    is_a = (sub_types == type_a)
    is_b = (sub_types == type_b)

    # Masked log-softmax
    score_a = sub_scores.masked_fill(~is_a, float('-inf'))
    score_b = sub_scores.masked_fill(~is_b, float('-inf'))

    log_prob_a = F.log_softmax(score_a, dim=-1)
    log_prob_b = F.log_softmax(score_b, dim=-1)

    # Find local indices from global swap_indices
    # swap_indices[:, 0] was type_a, swap_indices[:, 1] was type_b
    global_a = swap_indices[:, 0]  # [batch]
    global_b = swap_indices[:, 1]  # [batch]

    # Convert global to local (within sublattice)
    # sub_idx[local] = global → local = (sub_idx == global).argmax()
    local_a = (sub_idx.unsqueeze(0) == global_a.unsqueeze(1)).int().argmax(dim=-1)
    local_b = (sub_idx.unsqueeze(0) == global_b.unsqueeze(1)).int().argmax(dim=-1)

    # Gather log probs
    lp_a = log_prob_a[torch.arange(batch_size, device=device), local_a]
    lp_b = log_prob_b[torch.arange(batch_size, device=device), local_b]

    return lp_a + lp_b

@torch.no_grad()
def sample_sublattice_lazy_swap(
    atom_types: torch.Tensor,
    sublattice_mask: torch.Tensor,
    type_a: int,
    type_b: int,
    scores: Optional[torch.Tensor] = None,
    logit_unchanged: Optional[torch.Tensor] = None,
    deterministic: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Lazy swap: can choose to not swap (identity).
    Args:
        atom_types: [batch, N]
        sublattice_mask: [N]
        type_a, type_b: types to swap
        scores: [batch, N] swap scores
        logit_unchanged: [batch] logit for staying unchanged
        deterministic: if True, pick mode
    Returns:
        result: [batch, N] (possibly unchanged)
        indices: [batch, 2] (0, 0 if unchanged)
        changed: [batch] boolean
    """
    device = atom_types.device
    batch_size = atom_types.shape[0]

    if logit_unchanged is None:
        logit_unchanged = torch.zeros(batch_size, device=device)

    # Probability of staying unchanged
    prob_unchanged = torch.sigmoid(logit_unchanged)

    # Sample swap candidate
    swapped, indices = sample_sublattice_swap(
        atom_types, sublattice_mask, type_a, type_b, scores, deterministic
    )

    # Decide whether to swap or stay
    if deterministic:
        # Compare log probs
        log_unchanged = F.logsigmoid(logit_unchanged)
        log_changed = F.logsigmoid(-logit_unchanged)
        swap_log_prob = compute_single_swap_log_prob(
            scores, sublattice_mask, type_a, type_b, atom_types, indices
        )
        changed = (log_changed + swap_log_prob) > log_unchanged
    else:
        changed = torch.rand(batch_size, device=device) > prob_unchanged

    # Apply decision
    result = torch.where(changed.unsqueeze(-1), swapped, atom_types)
    final_indices = torch.where(
        changed.unsqueeze(-1),
        indices,
        torch.zeros_like(indices)
    )

    return result, final_indices, changed

def compute_single_swap_log_prob(
    scores: torch.Tensor,
    sublattice_mask: torch.Tensor,
    type_a: int,
    type_b: int,
    atom_types: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    """Helper to compute log prob for a single swap."""
    if scores is None:
        # Uniform: log(1/n_a) + log(1/n_b)
        sub_idx = torch.where(sublattice_mask)[0]
        sub_types = atom_types[:, sub_idx]
        n_a = (sub_types == type_a).sum(dim=-1).float()
        n_b = (sub_types == type_b).sum(dim=-1).float()
        return -torch.log(n_a) - torch.log(n_b)
    else:
        return log_prob_sublattice_swap(
            scores, sublattice_mask, type_a, type_b, atom_types, indices
        )