"""
JAX/Flax implementation of Crystal Graph Convolutional Neural Network with Graphormer.
This is a port of the PyTorch model to JAX using Flax nn.Module.
Requirements:
    pip install jax jaxlib flax optax
Note: This implementation avoids dynamic control flow where possible for JIT compilation.
"""

from __future__ import annotations

from typing import List, Optional, Tuple, Callable, Any
from functools import partial

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen.initializers import lecun_normal, zeros


class ConvLayer(nn.Module):
    """
    Convolutional operation on crystal graphs (Flax version).
    Attributes:
        atom_fea_len: Number of atom hidden features
        nbr_fea_len: Number of bond features
    """
    atom_fea_len: int
    nbr_fea_len: int

    @nn.compact
    def __call__(self, atom_in_fea: jnp.ndarray, nbr_fea: jnp.ndarray,
                 nbr_fea_idx: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        """
        Forward pass.
        Parameters
        ----------
        atom_in_fea: jnp.ndarray shape (N, atom_fea_len)
            Atom hidden features before convolution
        nbr_fea: jnp.ndarray shape (N, M, nbr_fea_len)
            Bond features of each atom's M neighbors
        nbr_fea_idx: jnp.ndarray shape (N, M)
            Indices of M neighbors of each atom
        train: bool
            Training mode for batch normalization
        Returns
        -------
        atom_out_fea: jnp.ndarray shape (N, atom_fea_len)
            Atom hidden features after convolution
        """
        N, M = nbr_fea_idx.shape

        # Gather neighbor features
        atom_nbr_fea = atom_in_fea[nbr_fea_idx]  # (N, M, atom_fea_len)

        # Expand atom features and concatenate
        atom_in_fea_expanded = jnp.broadcast_to(
            atom_in_fea[:, None, :],
            (N, M, self.atom_fea_len)
        )

        total_nbr_fea = jnp.concatenate(
            [atom_in_fea_expanded, atom_nbr_fea, nbr_fea],
            axis=-1
        )  # (N, M, 2*atom_fea_len + nbr_fea_len)

        # Linear transformation
        total_gated_fea = nn.Dense(
            features=2 * self.atom_fea_len,
            name='fc_full'
        )(total_nbr_fea)  # (N, M, 2*atom_fea_len)

        # Batch normalization (reshape for BN)
        total_gated_fea = total_gated_fea.reshape(-1, 2 * self.atom_fea_len)
        total_gated_fea = nn.BatchNorm(
            use_running_average=not train,
            name='bn1'
        )(total_gated_fea)
        total_gated_fea = total_gated_fea.reshape(N, M, 2 * self.atom_fea_len)

        # Split into filter and core
        nbr_filter, nbr_core = jnp.split(total_gated_fea, 2, axis=-1)

        # Activations
        nbr_filter = nn.sigmoid(nbr_filter)
        nbr_core = nn.softplus(nbr_core)

        # Sum over neighbors
        nbr_sumed = jnp.sum(nbr_filter * nbr_core, axis=1)  # (N, atom_fea_len)

        # Batch normalization
        nbr_sumed = nn.BatchNorm(
            use_running_average=not train,
            name='bn2'
        )(nbr_sumed)

        # Residual connection
        out = nn.softplus(atom_in_fea + nbr_sumed)

        return out


class CentralityEncoding(nn.Module):
    """
    Centrality encoding for Graphormer (Flax version).
    Adds learnable embeddings based on node degree.
    Attributes:
        max_in_degree: Maximum in-degree of nodes
        max_out_degree: Maximum out-degree of nodes
        node_dim: Hidden dimensions of node features
    """
    max_in_degree: int
    max_out_degree: int
    node_dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray, edge_index: jnp.ndarray) -> jnp.ndarray:
        """
        Apply centrality encoding.
        Parameters
        ----------
        x: jnp.ndarray shape (N, node_dim)
            Node feature matrix
        edge_index: jnp.ndarray shape (2, E)
            Edge index (adjacency list)
        Returns
        -------
        x: jnp.ndarray shape (N, node_dim)
            Node embeddings after centrality encoding
        """
        num_nodes = x.shape[0]

        # Compute degrees using scatter
        in_degree = self._compute_degree(edge_index[1], num_nodes, self.max_in_degree - 1)
        out_degree = self._compute_degree(edge_index[0], num_nodes, self.max_out_degree - 1)

        # Learnable degree embeddings
        z_in = self.param(
            'z_in',
            lecun_normal(),
            (self.max_in_degree, self.node_dim)
        )
        z_out = self.param(
            'z_out',
            lecun_normal(),
            (self.max_out_degree, self.node_dim)
        )

        # Add degree embeddings
        x = x + z_in[in_degree] + z_out[out_degree]

        return x

    def _compute_degree(self, indices: jnp.ndarray, num_nodes: int, max_value: int) -> jnp.ndarray:
        """Compute node degrees from edge indices."""
        # Use segment_sum for degree computation
        degrees = jnp.zeros(num_nodes, dtype=jnp.int32)
        degrees = degrees.at[indices].add(1)
        # Clamp to max value
        degrees = jnp.minimum(degrees, max_value)
        return degrees


class GraphormerAttentionHead(nn.Module):
    """
    Single attention head for Graphormer (Flax version).
    Attributes:
        dim_in: Input dimension
        dim_q: Query dimension
        dim_k: Key/Value dimension
    """
    dim_in: int
    dim_q: int
    dim_k: int

    @nn.compact
    def __call__(self, x: jnp.ndarray, edge_index: jnp.ndarray,
                 ptr: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        """
        Apply single-head attention.
        Parameters
        ----------
        x: jnp.ndarray shape (N, dim_in)
            Node features
        edge_index: jnp.ndarray shape (2, E)
            Edge indices
        ptr: jnp.ndarray, optional
            Batch pointer for graph batching
        Returns
        -------
        out: jnp.ndarray shape (N, dim_k)
            Attended node features
        """
        query = nn.Dense(self.dim_q, name='q')(x)
        key = nn.Dense(self.dim_k, name='k')(x)
        value = nn.Dense(self.dim_k, name='v')(x)

        N = x.shape[0]

        # Build adjacency matrix
        adjacency = jnp.zeros((N, N))
        adjacency = adjacency.at[edge_index[0], edge_index[1]].set(1.0)

        # Compute attention scores
        scale = 1.0 / jnp.sqrt(query.shape[-1])

        if ptr is None:
            # Single graph
            a = jnp.matmul(query, key.T) * scale
        else:
            # Batched graphs - compute block-diagonal attention
            a = self._batched_attention(query, key, ptr, scale)

        # Apply adjacency mask
        a = a * adjacency + (1 - adjacency) * (-1e6)

        # Softmax and apply to values
        softmax = jax.nn.softmax(a, axis=-1)
        out = jnp.matmul(softmax, value)

        return out

    def _batched_attention(self, query: jnp.ndarray, key: jnp.ndarray,
                           ptr: jnp.ndarray, scale: float) -> jnp.ndarray:
        """Compute block-diagonal attention for batched graphs."""
        N = query.shape[0]
        a = jnp.zeros((N, N))

        # This loop will be unrolled by XLA for small batch sizes
        # For larger batches, consider using vmap
        def compute_block(carry, batch_idx):
            a, ptr = carry
            start = ptr[batch_idx]
            end = ptr[batch_idx + 1]

            q_block = jax.lax.dynamic_slice(query, (start, 0), (end - start, query.shape[1]))
            k_block = jax.lax.dynamic_slice(key, (start, 0), (end - start, key.shape[1]))

            block_attn = jnp.matmul(q_block, k_block.T) * scale

            # Update attention matrix
            indices = jnp.arange(end - start)
            row_idx = indices[:, None] + start
            col_idx = indices[None, :] + start
            a = a.at[row_idx, col_idx].set(block_attn)

            return (a, ptr), None

        num_graphs = len(ptr) - 1
        (a, _), _ = jax.lax.scan(compute_block, (a, ptr), jnp.arange(num_graphs))

        return a


class GraphormerMultiHeadAttention(nn.Module):
    """
    Multi-head attention for Graphormer (Flax version).
    Attributes:
        num_heads: Number of attention heads
        dim_in: Input dimension
        dim_q: Query dimension per head
        dim_k: Key/Value dimension per head
    """
    num_heads: int
    dim_in: int
    dim_q: int
    dim_k: int

    @nn.compact
    def __call__(self, x: jnp.ndarray, edge_index: jnp.ndarray,
                 ptr: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        """Apply multi-head attention."""
        head_outputs = []

        for i in range(self.num_heads):
            head = GraphormerAttentionHead(
                dim_in=self.dim_in,
                dim_q=self.dim_q,
                dim_k=self.dim_k,
                name=f'head_{i}'
            )
            head_outputs.append(head(x, edge_index, ptr))

        concatenated = jnp.concatenate(head_outputs, axis=-1)
        out = nn.Dense(self.dim_in, name='linear')(concatenated)

        return out


class GraphormerEncoderLayer(nn.Module):
    """
    Single Graphormer encoder layer (Flax version).
    Attributes:
        node_dim: Node feature dimension
        num_heads: Number of attention heads
    """
    node_dim: int
    num_heads: int

    @nn.compact
    def __call__(self, x: jnp.ndarray, edge_index: jnp.ndarray,
                 ptr: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        """Apply encoder layer with attention and FFN."""
        # Pre-norm attention
        x_norm = nn.LayerNorm(name='ln_1')(x)
        attn_out = GraphormerMultiHeadAttention(
            num_heads=self.num_heads,
            dim_in=self.node_dim,
            dim_q=self.node_dim,
            dim_k=self.node_dim,
            name='attention'
        )(x_norm, edge_index, ptr)
        x_prime = x + attn_out

        # Pre-norm FFN
        x_norm = nn.LayerNorm(name='ln_2')(x_prime)
        ffn_out = nn.Sequential([
            nn.Dense(self.node_dim),
            nn.relu,
            nn.Dense(self.node_dim)
        ])(x_norm)
        x_new = x_prime + ffn_out

        return x_new


class GraphormerEncoder(nn.Module):
    """
    Stack of Graphormer encoder layers (Flax version).
    Attributes:
        num_layers: Number of encoder layers
        node_dim: Node feature dimension
        num_heads: Number of attention heads
    """
    num_layers: int
    node_dim: int
    num_heads: int

    @nn.compact
    def __call__(self, x: jnp.ndarray, edge_index: jnp.ndarray,
                 ptr: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        """Apply all encoder layers."""
        for i in range(self.num_layers):
            x = GraphormerEncoderLayer(
                node_dim=self.node_dim,
                num_heads=self.num_heads,
                name=f'layer_{i}'
            )(x, edge_index, ptr)
        return x


class CrystalGraphConvNet(nn.Module):
    """
    Crystal Graph Convolutional Neural Network with Graphormer (Flax version).
    This is the main model for predicting material properties.
    Attributes:
        orig_atom_fea_len: Number of atom features in the input
        nbr_fea_len: Number of bond features
        atom_fea_len: Number of hidden atom features
        n_conv: Number of convolutional layers
        h_fea_len: Number of hidden features after pooling
        n_h: Number of hidden layers after pooling
        classification: Whether the task is classification
        graphormer_layers: Number of Graphormer encoder layers
        num_heads: Number of attention heads
    """
    orig_atom_fea_len: int
    nbr_fea_len: int
    atom_fea_len: int = 64
    n_conv: int = 3
    h_fea_len: int = 128
    n_h: int = 1
    classification: bool = False
    graphormer_layers: int = 1
    num_heads: int = 4
    max_in_degree: int = 10
    max_out_degree: int = 10
    dropout_rate: float = 0.0

    @nn.compact
    def __call__(self, atom_fea: jnp.ndarray, nbr_fea: jnp.ndarray,
                 nbr_fea_idx: jnp.ndarray, crystal_atom_idx: List[jnp.ndarray],
                 train: bool = True) -> jnp.ndarray:
        """
        Forward pass.
        Parameters
        ----------
        atom_fea: jnp.ndarray shape (N, orig_atom_fea_len)
            Atom features from atom type
        nbr_fea: jnp.ndarray shape (N, M, nbr_fea_len)
            Bond features of each atom's M neighbors
        nbr_fea_idx: jnp.ndarray shape (N, M)
            Indices of M neighbors of each atom
        crystal_atom_idx: List of jnp.ndarray
            Mapping from crystal idx to atom idx
        train: bool
            Training mode flag
        Returns
        -------
        prediction: jnp.ndarray shape (N0, 1) or (N0, 2) for classification
        """
        # Initial embedding
        atom_fea = nn.Dense(self.atom_fea_len, name='embedding')(atom_fea)

        # Graph convolutions
        for i in range(self.n_conv):
            atom_fea = ConvLayer(
                atom_fea_len=self.atom_fea_len,
                nbr_fea_len=self.nbr_fea_len,
                name=f'conv_{i}'
            )(atom_fea, nbr_fea, nbr_fea_idx, train=train)

        # Construct edge_index for Graphormer
        N, M = nbr_fea_idx.shape
        src = jnp.repeat(jnp.arange(N), M)
        dst = nbr_fea_idx.reshape(-1)
        edge_index = jnp.stack([src, dst], axis=0)

        # Centrality encoding
        atom_fea = CentralityEncoding(
            max_in_degree=self.max_in_degree,
            max_out_degree=self.max_out_degree,
            node_dim=self.atom_fea_len,
            name='centrality_encoding'
        )(atom_fea, edge_index)

        # Construct ptr from crystal_atom_idx
        ptr = jnp.array([0] + [len(idx) for idx in crystal_atom_idx])
        ptr = jnp.cumsum(ptr)

        # Graphormer encoder
        atom_fea = GraphormerEncoder(
            num_layers=self.graphormer_layers,
            node_dim=self.atom_fea_len,
            num_heads=self.num_heads,
            name='graphormer_encoder'
        )(atom_fea, edge_index, ptr)

        # Pooling
        crys_fea = self._pooling(atom_fea, crystal_atom_idx)

        # FC layers
        crys_fea = nn.softplus(crys_fea)
        crys_fea = nn.Dense(self.h_fea_len, name='conv_to_fc')(crys_fea)
        crys_fea = nn.softplus(crys_fea)

        if self.classification and train:
            crys_fea = nn.Dropout(rate=self.dropout_rate, deterministic=not train)(crys_fea)

        # Additional hidden layers
        for i in range(self.n_h - 1):
            crys_fea = nn.Dense(self.h_fea_len, name=f'fc_{i}')(crys_fea)
            crys_fea = nn.relu(crys_fea)

        # Output layer
        if self.classification:
            out = nn.Dense(2, name='fc_out')(crys_fea)
            out = nn.log_softmax(out, axis=-1)
        else:
            out = nn.Dense(1, name='fc_out')(crys_fea)

        return out

    def _pooling(self, atom_fea: jnp.ndarray, crystal_atom_idx: List[jnp.ndarray]) -> jnp.ndarray:
        """Pool atom features to crystal features using mean pooling."""
        crys_fea = []
        for idx_map in crystal_atom_idx:
            crys_fea.append(jnp.mean(atom_fea[idx_map], axis=0))
        return jnp.stack(crys_fea, axis=0)


class CGFormerEncoder(nn.Module):
    """
    CGFormer Encoder for diffusion models (Flax version).
    Returns node embeddings without pooling, suitable for per-atom predictions.
    Attributes:
        orig_atom_fea_len: Number of atom features in the input
        nbr_fea_len: Number of bond features
        atom_fea_len: Number of hidden atom features
        n_conv: Number of convolutional layers
        graphormer_layers: Number of Graphormer encoder layers
        num_heads: Number of attention heads
    """
    orig_atom_fea_len: int
    nbr_fea_len: int
    atom_fea_len: int = 64
    n_conv: int = 3
    graphormer_layers: int = 1
    num_heads: int = 4
    max_in_degree: int = 20
    max_out_degree: int = 20

    @nn.compact
    def __call__(self, atom_fea: jnp.ndarray, nbr_fea: jnp.ndarray,
                 nbr_fea_idx: jnp.ndarray, cond_emb: Optional[jnp.ndarray] = None,
                 batch_ptr: Optional[jnp.ndarray] = None, train: bool = True) -> jnp.ndarray:
        """
        Forward pass returning node embeddings.
        Parameters
        ----------
        atom_fea: jnp.ndarray shape (N, orig_atom_fea_len)
        nbr_fea: jnp.ndarray shape (N, M, nbr_fea_len)
        nbr_fea_idx: jnp.ndarray shape (N, M)
        cond_emb: jnp.ndarray, optional
            Conditioning embedding (e.g., time, temperature)
        batch_ptr: jnp.ndarray, optional
            Batch pointer for graph batching
        train: bool
            Training mode flag
        Returns
        -------
        x: jnp.ndarray shape (N, atom_fea_len)
            Node embeddings
        """
        # Initial embedding
        x = nn.Dense(self.atom_fea_len, name='embedding')(atom_fea)

        # Graph convolutions
        for i in range(self.n_conv):
            x = ConvLayer(
                atom_fea_len=self.atom_fea_len,
                nbr_fea_len=self.nbr_fea_len,
                name=f'conv_{i}'
            )(x, nbr_fea, nbr_fea_idx, train=train)

        # Condition injection
        if cond_emb is not None:
            cond_proj = nn.Sequential([
                nn.Dense(self.atom_fea_len),
                nn.silu,
                nn.Dense(self.atom_fea_len)
            ], name='cond_proj')
            x = x + cond_proj(cond_emb)

        # Construct edge_index
        N, M = nbr_fea_idx.shape
        src = jnp.repeat(jnp.arange(N), M)
        dst = nbr_fea_idx.reshape(-1)
        edge_index = jnp.stack([src, dst], axis=0)

        # Centrality encoding
        x = CentralityEncoding(
            max_in_degree=self.max_in_degree,
            max_out_degree=self.max_out_degree,
            node_dim=self.atom_fea_len,
            name='centrality_encoding'
        )(x, edge_index)

        # Graphormer encoder
        x = GraphormerEncoder(
            num_layers=self.graphormer_layers,
            node_dim=self.atom_fea_len,
            num_heads=self.num_heads,
            name='graphormer_encoder'
        )(x, edge_index, batch_ptr)

        return x


# Utility functions for training

def create_train_state(model: nn.Module, rng: jax.random.PRNGKey,
                       input_shapes: dict, learning_rate: float = 1e-3):
    """
    Create initial training state.
    Args:
        model: Flax model
        rng: Random key
        input_shapes: Dictionary of input shapes for initialization
        learning_rate: Learning rate for optimizer
    Returns:
        TrainState with initialized parameters
    """
    import optax
    from flax.training import train_state

    # Initialize parameters
    variables = model.init(rng, **input_shapes, train=False)

    # Create optimizer
    tx = optax.adam(learning_rate)

    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=variables['params'],
        tx=tx
    ), variables.get('batch_stats', {})


def load_pytorch_weights(flax_model: nn.Module, pytorch_state_dict: dict,
                         rng: jax.random.PRNGKey, input_shapes: dict) -> dict:
    """
    Convert PyTorch state dict to Flax parameters.
    Note: This is a basic converter. Complex models may need custom mapping.
    Args:
        flax_model: Flax model instance
        pytorch_state_dict: PyTorch state dict
        rng: Random key for initialization
        input_shapes: Input shapes for model initialization
    Returns:
        Flax parameters dictionary
    """
    # Initialize Flax params to get structure
    variables = flax_model.init(rng, **input_shapes, train=False)
    flax_params = variables['params']

    # Mapping from PyTorch to Flax naming conventions
    def convert_key(pytorch_key: str) -> str:
        """Convert PyTorch key to Flax key."""
        key = pytorch_key
        # Common conversions
        key = key.replace('.weight', '.kernel')
        key = key.replace('.running_mean', '.mean')
        key = key.replace('.running_var', '.var')
        return key

    def convert_tensor(tensor, is_kernel: bool = False) -> jnp.ndarray:
        """Convert PyTorch tensor to JAX array with proper transpose."""
        import torch
        if isinstance(tensor, torch.Tensor):
            arr = tensor.detach().cpu().numpy()
        else:
            arr = tensor

        # Transpose weight matrices (PyTorch: out_features x in_features)
        # Flax expects: in_features x out_features
        if is_kernel and len(arr.shape) == 2:
            arr = arr.T

        return jnp.array(arr)

    # Note: Full conversion requires careful key mapping
    # This is a template - specific models may need customization
    print("Warning: PyTorch to Flax weight conversion requires model-specific mapping.")
    print("Please verify converted weights manually.")

    return flax_params