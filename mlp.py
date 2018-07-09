import torch
from torch import nn
from torch import autograd

class MLP(nn.Module):
    """
    Multilayer perceptron model.
    """

    def __init__(self, input_size, hidden_layers, output_size,
            dropout):
        super().__init__()

        layers = []

        sizes_before_output = [input_size] + hidden_layers
        for curr, next in zip(sizes_before_output, sizes_before_output[1:]):
            layers.append(nn.Linear(curr, next))
            layers.append(nn.ReLU())
            if dropout:
                layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(hidden_layers[-1], output_size))

        self.model = nn.Sequential(*layers)

    def forward(self, X):
        return self.model(X)
