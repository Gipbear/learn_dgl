import argparse

import dgl
import torch
import torch.nn as nn
from torch import Tensor
from dgl import DGLGraph, AddSelfLoop
from dgl.data import CiteseerGraphDataset, CoraGraphDataset, PubmedGraphDataset

from gnn_model import GCN


def evaluate(g: DGLGraph, features: Tensor, labels: Tensor, mask: Tensor, model: GCN):
    model.eval()  # 进入 评估模式，不启用 Dropout, BatchNorm 模式
    with torch.no_grad():  # 关闭梯度计算
        logits = model(g, features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)


def train(g: DGLGraph, features: Tensor, labels: Tensor, masks: tuple[Tensor], model: GCN):
    model.train()
    train_mask = masks[0]
    val_mask = masks[1]
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=5e-4)
    for epoch in range(200):
        model.train()
        logits = model(g, features)
        loss = loss_func(logits[train_mask], labels[train_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        acc = evaluate(g, features, labels, val_mask, model)
        print(
            "Epoch {:05d} | Loss {:.4f} | Accuracy {:.4f} ".format(
                epoch, loss.item(), acc
            )
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, default='cora', help="Dataset name ('cora', 'citeseer', 'pubmed')."
    )
    parser.add_argument(
        "--dt", type=str, default='float', help="data type(float, bfloat16)."
    )
    args = parser.parse_args()

    print(f"Training with DGL built-in GraphConv module.")
    transform = (
        AddSelfLoop()  # 为了避免重复，会先删除已有的 self-loops
    )
    if args.dataset == "cora":
        data = CoraGraphDataset(transform=transform)
    elif args.dataset == "citeseer":
        data = CiteseerGraphDataset(transform=transform)
    elif args.dataset == "pubmed":
        data = PubmedGraphDataset(transform=transform)
    else:
        raise ValueError("Unknown dataset: {}".format(args.dataset))

    g = data[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    g = g.int().to(device)
    features = g.ndata['feat']
    labels = g.ndata['label']
    masks = g.ndata['train_mask'], g.ndata['val_mask'], g.ndata['test_mask']
    # 创建 GCN 模型
    in_size = features.shape[1]
    out_size = data.num_classes
    model = GCN(in_size, 16, out_size).to(device)
    if args.dt == "bfloat16":
        g = dgl.to_bfloat16(g)
        features = features.to(dtype=torch.bfloat16)
        model = model.to(dtype=torch.bfloat16)

    print(model)
    for param in model.parameters():
        print(type(param.data), param.size())
    print("Training...")
    train(g, features, labels, masks, model)
    print("Testing...")
    acc = evaluate(g, features, labels, masks[2], model)
    print("Test accuracy {:.4f}".format(acc))
