from typing import Callable

import torch

from implementations.linear_layer import linear_layer
from utils.extended_matcher import replace_pattern


def linear_wrapper(v, linear, activation=""):
    if not hasattr(linear, "weight_h"):
        linear.weight_h = linear.weight.half()
        linear.weight = None
    if not hasattr(linear, "bias_h") and linear.bias is not None:
        linear.bias_h = linear.bias.half()
        linear.bias = None

    return linear_layer(v, linear.weight_h, linear.bias_h if hasattr(linear, "bias_h") is not None else None, activation=activation)


torch.fx.wrap('linear_wrapper')


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
            return linear_wrapper(v, self.linear, activation=activation)

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
            return linear_wrapper(v, self.linear)

    replace_pattern(gm, Pattern(), Replacement())


def replace_all_linear(gm: torch.fx.GraphModule):
    replace_linear_activation(gm, torch.nn.Tanh(), "tanh")
    replace_linear_activation(gm, torch.nn.ReLU(), "relu")
    replace_linear_activation(gm, torch.nn.functional.gelu, "gelu")
    replace_linear(gm)
