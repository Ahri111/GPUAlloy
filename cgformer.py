import torch
import torch.nn as nn
from torch_geometric.utils import degree

"""
Atom_fea: [Number of atoms in unit cell, 92 (periodic table)] -> based on atom_init.json
nbr_fea_idx: (number of atoms in unit cell, max_num_nbr) -> which atoms around the central atom
nbr_fea: (number of atoms in unit cell, max_num_nbr, gaussian expansion) -> distance to the neighbor atoms 
"""


class ConvLayer(nn.Module):
    def __init__(self, atom_fea_len, nbr_fea_len):
        super(ConvLayer, self).__init__()
        self.atom_fea_len = atom_fea_len
        self.nbr_fea_len = nbr_fea_len
        self.fc_full = nn.Linear(2 * self.atom_fea_len + self.nbr_fea_len, 2 * self.atom_fea_len)
        self.sigmoid = nn.Sigmoid()
        self.softplus1 = nn.Softplus()
        self.bn1 = nn.BatchNorm1d(2 * self.atom_fea_len)
        self.bn2 = nn.BatchNorm1d(self.atom_fea_len)
        self.softplus2 = nn.Softplus()

    def forward(self, atom_in_fea, nbr_fea, nbr_fea_idx):
        N, M = nbr_fea_idx.shape
        atom_nbr_fea = atom_in_fea[nbr_fea_idx, :]
        total_nbr_fea = torch.cat(
            [atom_in_fea.unsqueeze(1).expand(N, M, self.atom_fea_len),
             atom_nbr_fea, nbr_fea], dim=2)
        total_gated_fea = self.fc_full(total_nbr_fea)
        total_gated_fea = self.bn1(total_gated_fea.view(
            -1, self.atom_fea_len * 2)).view(N, M, self.atom_fea_len * 2)
        nbr_filter, nbr_core = total_gated_fea.chunk(2, dim=2)
        nbr_filter = self.sigmoid(nbr_filter)
        nbr_core = self.softplus1(nbr_core)
        nbr_sumed = torch.sum(nbr_filter * nbr_core, dim=1)
        nbr_sumed = self.bn2(nbr_sumed)
        out = self.softplus2(atom_in_fea + nbr_sumed)
        return out

# --- [Copy from taokehao/model.py] CentralityEncoding ---
class CentralityEncoding(nn.Module):
    def __init__(self, max_in_degree: int, max_out_degree: int, node_dim: int):
        super().__init__()
        self.max_in_degree = max_in_degree
        self.max_out_degree = max_out_degree
        self.node_dim = node_dim
        self.z_in = nn.Parameter(torch.randn((max_in_degree, node_dim)))
        self.z_out = nn.Parameter(torch.randn((max_out_degree, node_dim)))

    def forward(self, x: torch.Tensor, edge_index: torch.LongTensor) -> torch.Tensor:
        num_nodes = x.shape[0]
        in_degree = self.decrease_to_max_value(degree(index=edge_index[1], num_nodes=num_nodes).long(),
                                               self.max_in_degree - 1)
        out_degree = self.decrease_to_max_value(degree(index=edge_index[0], num_nodes=num_nodes).long(),
                                                self.max_out_degree - 1)
        x = x + self.z_in[in_degree] + self.z_out[out_degree]
        return x

    def decrease_to_max_value(self, x, max_value):
        return torch.clamp(x, max=max_value)

class GraphormerAttentionHead(nn.Module):
    def __init__(self, dim_in: int, dim_q: int, dim_k: int):
        super().__init__()
        self.q = nn.Linear(dim_in, dim_q)
        self.k = nn.Linear(dim_in, dim_k)
        self.v = nn.Linear(dim_in, dim_k)

    def forward(self, x: torch.Tensor, edge_index: torch.LongTensor, ptr=None) -> torch.Tensor:
        query = self.q(x)
        key = self.k(x)
        value = self.v(x)
        
        # [NOTE] Global Attention을 위해 Adjacency Masking 부분 수정 가능
        # 여기서는 원본 유지하되, 필요시 adjacency 제약을 풀 수 있음.
        if edge_index is not None:
            N = x.size(0)
            adjacency = torch.zeros(N, N, device=x.device)
            adjacency[edge_index[0], edge_index[1]] = 1.0
        else:
            # edge_index가 없으면 Fully Connected로 가정
            adjacency = 1.0

        if ptr is None:
            a = query.mm(key.transpose(0, 1)) / (query.size(-1) ** 0.5)
        else:
            a = torch.zeros((query.shape[0], query.shape[0]), device=x.device)
            for i in range(len(ptr) - 1):
                a[ptr[i]:ptr[i + 1], ptr[i]:ptr[i + 1]] = query[ptr[i]:ptr[i + 1]].mm(
                    key[ptr[i]:ptr[i + 1]].transpose(0, 1)) / (query.size(-1) ** 0.5)

        # Apply adjacency mask logic (Original Logic preserved)
        if isinstance(adjacency, torch.Tensor):
            a = a * adjacency + (1 - adjacency) * (-1e6)
        
        softmax = torch.softmax(a, dim=-1)
        out = softmax.mm(value)
        return out

class GraphormerMultiHeadAttention(nn.Module):
    def __init__(self, num_heads: int, dim_in: int, dim_q: int, dim_k: int):
        super().__init__()
        self.heads = nn.ModuleList(
            [GraphormerAttentionHead(dim_in, dim_q, dim_k) for _ in range(num_heads)]
        )
        self.linear = nn.Linear(num_heads * dim_k, dim_in)

    def forward(self, x: torch.Tensor, edge_index: torch.LongTensor, ptr) -> torch.Tensor:
        head_outs = []
        for attention_head in self.heads:
            head_out = attention_head(x, edge_index, ptr)
            head_outs.append(head_out)
        concatenated = torch.cat(head_outs, dim=-1)
        out = self.linear(concatenated)
        return out

class GraphormerEncoderLayer(nn.Module):
    def __init__(self, node_dim, num_heads):
        super().__init__()
        self.attention = GraphormerMultiHeadAttention(
            num_heads=num_heads,
            dim_in=node_dim,
            dim_q=node_dim,
            dim_k=node_dim,
        )
        self.ln_1 = nn.LayerNorm(node_dim)
        self.ln_2 = nn.LayerNorm(node_dim)
        self.ff = nn.Sequential(
            nn.Linear(node_dim, node_dim),
            nn.ReLU(),
            nn.Linear(node_dim, node_dim)
        )

    def forward(self, x: torch.Tensor, edge_index: torch.LongTensor, ptr) -> torch.Tensor:
        x_prime = self.attention(self.ln_1(x), edge_index, ptr) + x
        x_new = self.ff(self.ln_2(x_prime)) + x_prime
        return x_new

class GraphormerEncoder(nn.Module):
    def __init__(self, layers, node_dim, num_heads):
        super().__init__()
        self.layers = nn.ModuleList([
            GraphormerEncoderLayer(node_dim, num_heads) for _ in range(layers)
        ])

    def forward(self, x, edge_index, ptr):
        for layer in self.layers:
            x = layer(x, edge_index, ptr)
        return x

class CGFormerEncoder(nn.Module):
    def __init__(self, orig_atom_fea_len, nbr_fea_len,
                 atom_fea_len=64, n_conv=3, graphormer_layers=1, num_heads=4):
        super(CGFormerEncoder, self).__init__()
        
        # [Original] Embedding & ConvLayers
        self.embedding = nn.Linear(orig_atom_fea_len, atom_fea_len)
        self.convs = nn.ModuleList([ConvLayer(atom_fea_len=atom_fea_len,
                                              nbr_fea_len=nbr_fea_len)
                                    for _ in range(n_conv)])
        
        # [Original] Centrality & Graphormer
        self.centrality_encoding = CentralityEncoding(max_in_degree=20, max_out_degree=20, node_dim=atom_fea_len)
        self.graphormer_encoder = GraphormerEncoder(
            layers=graphormer_layers,
            node_dim=atom_fea_len,
            num_heads=num_heads
        )
        
        # [New] Condition Injection Layer (Time + Temperature)
        # Diffusion Step 및 Temperature 정보를 주입하기 위해 새로 추가된 층
        self.cond_proj = nn.Sequential(
            nn.Linear(atom_fea_len, atom_fea_len),
            nn.SiLU(),
            nn.Linear(atom_fea_len, atom_fea_len)
        )
        
        # [Removed] Pooling Layers & FC Output
        # 원본의 self.conv_to_fc, self.fc_out 등은 제거됨 (Node Embedding 반환 목적)

    def forward(self, atom_fea, nbr_fea, nbr_fea_idx, cond_emb=None, batch_ptr=None):
        # 1. Init Embedding
        x = self.embedding(atom_fea) # [N, atom_fea_len]
        
        # 2. Local Conv
        for conv_func in self.convs:
            x = conv_func(x, nbr_fea, nbr_fea_idx)
            
        # [NEW] Inject Condition (Time/Temp)
        if cond_emb is not None:
            x = x + self.cond_proj(cond_emb)

        # 3. Global Attention & Centrality
        N, M = nbr_fea_idx.shape
        src = torch.repeat_interleave(torch.arange(N, device=x.device), M)
        dst = nbr_fea_idx.view(-1)
        edge_index = torch.stack([src, dst], dim=0)

        x = self.centrality_encoding(x, edge_index)
        x = self.graphormer_encoder(x, edge_index, batch_ptr)
        
        # Pooling 없이 Node Embedding 반환
        return x