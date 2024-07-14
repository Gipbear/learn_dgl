import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, n_input: int, n_hidden: int, n_output: int, n_layers: int) -> None:
        assert n_layers >= 3
        super().__init__(n_input, n_hidden, n_output, n_layers)
        self.n_layers = n_layers
        self.linear_layers = nn.ModuleList()
        self.linear_layers.append(nn.Linear(n_input, n_hidden))
        for _ in range(n_layers - 2):
            self.linear_layers.append(nn.Linear(n_hidden, n_hidden))
        self.linear_layers.append(nn.Linear(n_hidden, n_output))

        self.bn_layers = nn.ModuleList()
        for _ in range(n_layers - 1):
            self.bn_layers.append(nn.BatchNorm1d(n_hidden))

    def forward(self, features):
        x = features
        for i in range(self.n_layers):
            x = self.linear_layers[i](x)
            x = self.bn_layers[i](x)
        x = self.linear_layers[-1](x)
        return x
