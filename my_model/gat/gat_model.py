import dgl.nn as dglnn
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph
from torch import Tensor


class GAT(nn.Module):
    def __init__(self, in_size, hid_size, out_size, heads) -> None:
        super().__init__()
        self.gat_layers = nn.ModuleList()
        # 两层 GAT
        self.gat_layers.append(
            dglnn.GATConv(in_size, hid_size, heads[0], feat_drop=0.6, attn_drop=0.6, activation=F.elu)
        )
        self.gat_layers.append(
            dglnn.GATConv(hid_size * heads[0], out_size, heads[1], feat_drop=0.6, attn_drop=0.6, activation=None)
        )

    def forward(self, g: DGLGraph, features: Tensor):
        h = features
        for i, layer in enumerate(self.gat_layers):
            h = layer(g, h)
            if i == len(self.gat_layers) - 1:  # 最后一层
                h = h.mean(1)
            else:
                h = h.flatten(1)
        return h
