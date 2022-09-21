from typing import Callable

import torch

from implementations.linear_layer import linear_layer
from utils.extended_matcher import replace_pattern

torch.fx.wrap('linear_layer')


def replace_linear_activation(gm: torch.fx.GraphModule, activation_module: Callable, activation: str):
    class Pattern(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(1, 1)
            self.activation = activation_module

        def forward(self, v):
            return self.activation(self.linear(v))

    class Replacement(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(1, 1)
            self.activation = activation_module

        def forward(self, v):
            linear = linear_layer(v, self.linear.weight.data, self.linear.bias.data, activation=activation)
            out = linear[0]
            return out

    replace_pattern(gm, Pattern(), Replacement())


def replace_linear(gm: torch.fx.GraphModule):
    class Pattern(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(1, 1)

        def forward(self, v):
            return self.linear(v)

    class Replacement(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(1, 1)

        def forward(self, v):
            output = linear_layer(v, self.linear.weight.data, self.linear.bias.data)
            return output

    replace_pattern(gm, Pattern(), Replacement())


def replace_all_linear(gm: torch.fx.GraphModule):
    replace_linear_activation(gm, torch.nn.Tanh(), "tanh")
    replace_linear_activation(gm, torch.nn.functional.gelu, "gelu")
    replace_linear(gm)
