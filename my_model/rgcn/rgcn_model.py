import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph
from dgl.nn.pytorch import RelGraphConv


class RGCN(nn.Module):
    def __init__(self, num_nodes, h_dim, out_dim, num_rels) -> None:
        super().__init__()
        self.emb = nn.Embedding(num_nodes, h_dim)
        self.conv1 = RelGraphConv(
            h_dim, h_dim, num_rels, regularizer='basis', num_bases=num_rels, self_loop=False
        )
        self.conv2 = RelGraphConv(
            h_dim, out_dim, num_rels, regularizer='basis', num_bases=num_rels, self_loop=False
        )

    def forward(self, g: DGLGraph):
        x = self.emb.weight
        h = F.relu(self.conv1(g, x, g.edata[dgl.ETYPE], g.edata["norm"]))
        h = self.conv2(g, h, g.edata[dgl.ETYPE], g.edata["norm"])
        return h
