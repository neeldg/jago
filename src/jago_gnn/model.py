"""GraphSAGE-style GNNs for JAGO, implemented in pure PyTorch (no PyG).

SAGELayer — one mean-aggregation message-passing step.
MaskedCellGNN — node-level cell-type classifier (masked-cell task).
NeighborhoodCompletionGNN — graph-level encoder for neighborhood completion:
    encodes the context graph, optionally using a virtual node for global
    aggregation, then predicts cell-type composition fractions and log1p
    cell count inside the hidden circular region.
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


def _mean_pool(h: torch.Tensor, batch_index: torch.Tensor, n_graphs: int) -> torch.Tensor:
    """Scatter-mean: average node embeddings within each graph in a batched graph."""
    out = torch.zeros(n_graphs, h.size(1), device=h.device, dtype=h.dtype)
    out.index_add_(0, batch_index, h)
    counts = torch.bincount(batch_index, minlength=n_graphs).clamp(min=1).to(h.dtype).unsqueeze(1)
    return out / counts


class NeighborhoodCompletionGNN(nn.Module):
    """Graph-level encoder for the neighborhood completion task.

    Encodes the context graph (cells outside the circular mask) and predicts:
      - composition logits (C outputs; apply softmax for cell-type fractions)
      - count scalar (target is log1p(n_hidden_cells))

    When use_virtual_node=True, a learnable virtual node is appended to each
    graph and connected bidirectionally to every real node before message
    passing.  Its embedding after the final layer becomes the graph
    representation, allowing long-range information flow without extra depth.
    When False, plain mean pooling over real nodes is used instead.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_classes: int,
        num_layers: int = 3,
        dropout: float = 0.1,
        use_virtual_node: bool = True,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")

        self.use_virtual_node = use_virtual_node
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.layers = nn.ModuleList([SAGELayer(hidden_dim, hidden_dim) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)

        if use_virtual_node:
            # One learnable initialisation shared across all virtual nodes;
            # each virtual node diverges during message passing.
            self.virtual_init = nn.Parameter(torch.zeros(1, hidden_dim))

        self.composition_head = nn.Linear(hidden_dim, num_classes)
        self.count_head = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch_index: torch.Tensor,
        n_graphs: int,
    ) -> tuple:
        """Return (comp_logits [G, C], count_pred [G])."""
        n_real = x.size(0)
        h = F.relu(self.input_proj(x))

        if self.use_virtual_node:
            v = self.virtual_init.expand(n_graphs, -1)  # (G, hidden_dim)
            h = torch.cat([h, v], dim=0)               # (N+G, hidden_dim)

            real_nodes = torch.arange(n_real, device=x.device)
            virt_per_node = batch_index + n_real        # virtual node index for each real node
            vn_edges = torch.stack([
                torch.cat([real_nodes, virt_per_node]),
                torch.cat([virt_per_node, real_nodes]),
            ])
            aug_edge_index = (
                torch.cat([edge_index, vn_edges], dim=1) if edge_index.numel() > 0 else vn_edges
            )
        else:
            aug_edge_index = edge_index

        for layer in self.layers:
            h = F.relu(layer(h, aug_edge_index))
            h = self.dropout(h)

        if self.use_virtual_node:
            graph_embed = h[n_real:]                    # (G, hidden_dim)
        else:
            graph_embed = _mean_pool(h, batch_index, n_graphs)

        comp_logits = self.composition_head(graph_embed)          # (G, C)
        count_pred = self.count_head(graph_embed).squeeze(-1)     # (G,)
        return comp_logits, count_pred
