from torch import Tensor
from dgl import DGLGraph
import torch.nn.functional as F
import torch.nn as nn
import dgl.nn as dglnn


class GAT(nn.Module):
    def __init__(self, in_size: int, hid_size: int, out_size: int, n_head: int, n_layer: int = 3) -> None:
        assert n_layer >= 3
        super().__init__()
        self.input_drop_rate = 0.1
        self.input_drop = nn.Dropout(self.input_drop_rate)
        self.gat_layers = nn.ModuleList()
        self.gat_layers.append(
            dglnn.GATConv(in_size, hid_size, n_head, feat_drop=0.1, attn_drop=0.1, residual=True, activation=None)
        )
        [self.gat_layers.append(
            dglnn.GATConv(hid_size * n_head, hid_size, n_head, feat_drop=0.3,
                          residual=True, attn_drop=0.1, activation=None)
        ) for _ in range(n_layer - 2)]
        self.gat_layers.append(
            dglnn.GATConv(hid_size * n_head, out_size, n_head, feat_drop=0.3,
                          residual=True, attn_drop=0.1, activation=F.relu)
        )

    def forward(self, g: DGLGraph, features: Tensor):
        h = self.input_drop(features)
        for i, layer in enumerate(self.gat_layers):
            h = layer(g, h)  # node_num x head_num x out_num
            if i == len(self.gat_layers) - 1:
                h = h.mean(axis=1)
            else:
                h = h.flatten(start_dim=1)  # 展平 heads
        return h


class GCN(nn.Module):
    def __init__(self, in_size, hid_size, out_size) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        # 两层 GCN
        self.layers.append(
            dglnn.GraphConv(in_size, hid_size, residual=True, activation=F.relu)
        )
        self.layers.append(
            dglnn.GraphConv(hid_size, hid_size, residual=True, activation=F.relu)
        )
        self.layers.append(
            dglnn.GraphConv(hid_size, out_size)
        )
        self.dropout = nn.Dropout(p=0.3)  # 前向传播时，神经元以概率 p 停止工作，增强模型泛化性

    def forward(self, g, features):
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(g, h)
        return h


class GAT_V2(nn.Module):
    def __init__(self, in_size: int, hid_size: int, out_size: int, heads: list[int]) -> None:
        super().__init__()
        self.gat_layers = nn.ModuleList()
        self.gat_layers.append(
            dglnn.GATv2Conv(in_size, hid_size, heads[0], feat_drop=0.3, attn_drop=0.1, residual=True, activation=F.relu)
        )
        self.gat_layers.append(
            dglnn.GATv2Conv(hid_size * heads[0], hid_size, heads[1], feat_drop=0.3,
                            residual=True, attn_drop=0.1, activation=F.relu)
        )
        self.gat_layers.append(
            dglnn.GATv2Conv(hid_size * heads[1], out_size, heads[2], feat_drop=0.3,
                            residual=True, attn_drop=0.1, activation=None)
        )

    def forward(self, g: DGLGraph, features: Tensor):
        h = features
        for i, layer in enumerate(self.gat_layers):
            h = layer(g, h)  # node_num x head_num x out_num
            if i == len(self.gat_layers) - 1:
                h = h.mean(axis=1)
            else:
                h = h.flatten(start_dim=1)  # 展平 heads
        return h
