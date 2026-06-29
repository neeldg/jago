"""A minimal GraphSAGE-style GNN, implemented in pure PyTorch (no PyG).

Each layer aggregates each node's neighbor features by mean-pooling (using
`index_add_` over the edge list) and combines that with the node's own
features via two separate linear maps, following the original GraphSAGE
"mean aggregator" formulation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SAGELayer(nn.Module):
    """One mean-aggregation GraphSAGE layer.

    Expects `edge_index` to already be symmetrized (both (src, dst) and
    (dst, src) present for every undirected edge) so that aggregating over
    `dst` collects every node's full neighborhood.
    """

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.lin_self = nn.Linear(in_dim, out_dim)
        self.lin_neigh = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        n_nodes = x.size(0)

        if edge_index.numel() == 0:
            neigh_mean = torch.zeros_like(x)
        else:
            src, dst = edge_index[0], edge_index[1]

            neigh_sum = torch.zeros(n_nodes, x.size(1), device=x.device, dtype=x.dtype)
            neigh_sum.index_add_(0, dst, x[src])

            counts = torch.zeros(n_nodes, device=x.device, dtype=x.dtype)
            ones = torch.ones(src.size(0), device=x.device, dtype=x.dtype)
            counts.index_add_(0, dst, ones)
            counts = counts.clamp(min=1.0).unsqueeze(1)

            neigh_mean = neigh_sum / counts

        return self.lin_self(x) + self.lin_neigh(neigh_mean)


class MaskedCellGNN(nn.Module):
    """Node-level classifier: predicts cell type at every node.

    Loss/metrics should only be computed at masked-node positions (the
    surrounding visible cell types, coordinates, and graph structure are
    what the model uses to infer the masked label).
    """

    def __init__(self, in_dim: int, hidden_dim: int, num_classes: int, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")

        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.layers = nn.ModuleList([SAGELayer(hidden_dim, hidden_dim) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.input_proj(x))
        for layer in self.layers:
            h = F.relu(layer(h, edge_index))
            h = self.dropout(h)
        return self.classifier(h)
