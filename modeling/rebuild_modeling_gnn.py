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
    def __init__(self, emb_dim, n_ntype, n_etype, edge_encoder, head_count=4, aggr="add"):
        super(GATConvE, self).__init__(aggr=aggr)

        # emb_dim must be even
        assert emb_dim % 2 == 0

        self.emb_dim = emb_dim
        self.n_ntype = n_ntype
        self.n_etype = n_etype
        self.edge_encoder = edge_encoder

        # For Attention
        self.head_count = head_count
        assert emb_dim % head_count == 0


    raise NotImplementedError