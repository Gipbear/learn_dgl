import dgl.nn as dglnn
import torch.nn as nn
import torch.nn.functional as F


class GCN(nn.Module):
    def __init__(self, in_size, hid_size, out_size) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        # 两层 GCN
        self.layers.append(
            dglnn.GraphConv(in_size, hid_size, activation=F.relu)
        )
        self.layers.append(
            dglnn.GraphConv(hid_size, out_size)
        )
        self.dropout = nn.Dropout(p=0.5)  # 前向传播时，神经元以概率 p 停止工作，增强模型泛化性

    def forward(self, g, features):
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(g, h)
        return h
