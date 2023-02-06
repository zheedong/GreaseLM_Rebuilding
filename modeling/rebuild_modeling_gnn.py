import math

import torch
from torch.autograd import Variable
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax
from torch_scatter import scatter

from einops import rearrange


def make_one_hot(labels, C):
    labels = rearrange(labels, 'n -> n 1')
    one_hot = torch.zeros(labels.size(0), C).to(labels.device)
    # dim, index, src. Now one_hot only has 0.
    # index[0][i] = 3, then one_hot[0][3] = 1
    # index[1][i] = 4, then one_hot[1][4] = 1
    # Continue...
    target = one_hot.scatter_(1, labels, 1)
    return target


class GATConvE(MessagePassing):
    def __init__(
            self,
            emb_dim,
            n_ntype,
            n_etype,
            edge_encoder,
            head_count=4,
            aggr="add"):
        super(GATConvE, self).__init__(aggr=aggr)

        # emb_dim must be even
        assert emb_dim % 2 == 0

        self.emb_dim = emb_dim
        # number of node types. {Z, Q, A, O}
        self.n_ntype = n_ntype
        # number of edge relation. 38
        self.n_etype = n_etype
        self.edge_encoder = edge_encoder

        # For Attention
        self.head_count = head_count
        assert emb_dim % head_count == 0
        self.dim_per_head = emb_dim // head_count
        self.linear_key = nn.Linear(
            3 * emb_dim, head_count * self.dim_per_head)
        self.linear_msg = nn.Linear(
            3 * emb_dim, head_count * self.dim_per_head)
        self.linear_query = nn.Linear(
            2 * emb_dim, head_count * self.dim_per_head)

        self._alpha = None

        # For Final MLP
        self.mlp = nn.Sequential(torch.nn.Linear(emb_dim, emb_dim),
                                 torch.nn.BatchNorm1d(emb_dim),
                                 torch.nn.ReLU(),
                                 torch.nn.Linear(emb_dim, emb_dim))

    def forward(
            self,
            x,
            edge_index,
            edge_type,
            node_type,
            node_feature_extra,
            return_attention_weights=False):
        # Prepare edge features

        # Why + 1? Maybe because of the interaction token?
        edge_vec = make_one_hot(edge_type, self.n_etype + 1)
        self_edge_vec = torch.zeros(
            x.size(0),
            self.n_etype +
            1).to(
            edge_vec.device)
        # TODO : Understand this. Why??
        self_edge_vec[:, self.n_etype] = 1

        head_type = node_type[edge_index[0]]
        tail_type = node_type[edge_index[1]]
        head_vec = make_one_hot(head_type, self.n_ntype)
        tail_vec = make_one_hot(tail_type, self.n_ntype)
        headtail_vec = torch.cat([head_vec, tail_vec], dim=1)
        self_head_vec = make_one_hot(node_type, self.n_ntype)
        self_headtail_vec = torch.cat([self_head_vec, self_head_vec], dim=1)

        edge_vec = torch.cat([edge_vec, self_edge_vec], dim=0)
        headtail_vec = torch.cat([headtail_vec, self_headtail_vec], dim=0)
        edge_embeddings = self.edge_encoder(
            torch.cat([edge_vec, headtail_vec], dim=1))

    raise NotImplementedError
