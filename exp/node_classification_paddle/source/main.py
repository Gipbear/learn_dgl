import argparse

import dgl
import torch
import torch.nn as nn
from torch import Tensor
from dgl import DGLGraph, AddSelfLoop

from model.gnn_model import GAT, GCN, GAT_V2
from util.load_data import ArixvDataSet


def evaluate(g: DGLGraph, features: Tensor, labels: Tensor, mask: Tensor, model):
    model.eval()
    with torch.no_grad():
        logits = model(g, features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)


def train(g: DGLGraph, features: Tensor, labels: Tensor, masks: tuple[Tensor], model, num_epochs: int):
    train_mask = masks[0]
    val_mask = masks[1]
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3, weight_decay=5e-4)
    for epoch in range(num_epochs):
        model.train()
        logits = model(g, features)
        loss = loss_func(logits[train_mask], labels[train_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        acc = evaluate(g, features, labels, val_mask, model)
        print(f"Epoch {epoch:05d} | loss {loss.item():.4f} | Accuracy {acc:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_gpus", type=int, default=0, help="Number of GPUs used for train and evaluation."
    )
    parser.add_argument(
        "--hidden_num", type=int, default=128, help="number of hidden layer"
    )
    parser.add_argument(
        "--num_epochs", type=int, default=200, help="Number of epochs for train."
    )
    args = parser.parse_args()

    print(f"Training with DGL built-in GATConv module.")
    transform = AddSelfLoop()
    data = ArixvDataSet(name='arixv', raw_dir='data', transform=transform)
    graph = data[0]
    print(graph)
    if args.num_gpus > 0 and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    graph = graph.int().to(device)
    features = graph.ndata['feat']
    labels = graph.ndata['label']
    masks = graph.ndata['train_mask'], graph.ndata['val_mask'], graph.ndata['test_mask']
    in_size = features.shape[1]
    out_size = graph.num_classes
    # model = GAT(in_size, args.hidden_num, out_size, heads=[2, 2, 1]).to(device)
    # model = GCN(in_size, args.hidden_num, out_size).to(device)
    model = GAT_V2(in_size, args.hidden_num, out_size, heads=[3, 3, 1]).to(device)
    print(model)
    for param in model.parameters():
        print(type(param.data), param.size())
    print("Training...")
    train(graph, features, labels, masks, model, args.num_epochs)
    print("Testing...")
    acc = evaluate(graph, features, labels, masks[2], model)
    print(f"Test accuracy {acc:.4f}")
