# The R-GCN model (PyTorch Geometric).
# Each layer: one message per relation (mean of the neighbours, own weight matrix per relation),
# the messages are mixed with softmax gates (learned weights that sum to 1), plus a self-loop.
# Two layers, then a linear head that gives one number per node.

import torch
import torch.nn as nn
from torch_geometric.nn import SimpleConv


class RGCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, n_relations):
        super().__init__()
        self.self_loop = nn.Linear(in_dim, out_dim)
        self.rel_weights = nn.ModuleList([nn.Linear(in_dim, out_dim, bias=False) for _ in range(n_relations)])
        self.mean_of_neighbours = SimpleConv(aggr="mean")      # PyG message passing, no own weights
        # one gate value per relation, softmax makes them positive and summing to 1
        self.gate_logits = nn.Parameter(torch.zeros(n_relations))

    def forward(self, h, edge_indexes):
        gates = torch.softmax(self.gate_logits, dim=0)
        out = self.self_loop(h)
        for r, edge_index in enumerate(edge_indexes):
            message = self.mean_of_neighbours(self.rel_weights[r](h), edge_index)   # relation r
            out = out + gates[r] * message
        return out


class RGCN(nn.Module):
    def __init__(self, in_dim, n_relations, hidden=256, dropout=0.4):
        super().__init__()
        self.layer1 = RGCNLayer(in_dim, hidden, n_relations)
        self.layer2 = RGCNLayer(hidden, hidden, n_relations)
        self.head = nn.Linear(hidden, 1)
        # start with output 0 (= no correction), so an untrained model gives the interpolation
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_indexes):
        h = self.dropout(torch.relu(self.layer1(x, edge_indexes)))
        h = self.dropout(torch.relu(self.layer2(h, edge_indexes)))
        return self.head(h).squeeze(-1)

    def relation_weights(self):
        # the learned softmax gates of both layers (useful to see which relations matter)
        return [torch.softmax(l.gate_logits, 0).detach().numpy() for l in (self.layer1, self.layer2)]
