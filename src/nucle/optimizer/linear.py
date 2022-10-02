from typing import Callable

import torch

from nucle.implementations.linear_layer import linear_layer
from nucle.utils.extended_matcher import replace_pattern


def linear_wrapper(v: torch.Tensor, linear: torch.nn.Linear, activation=""):
    # small hack to avoid casting weights/bias at each call
    if linear.weight.dtype == torch.float32:
        linear.weight.data = linear.weight.data.half()
    if linear.bias is not None and linear.bias.dtype == torch.float32:
        linear.bias.data = linear.bias.data.half()

    return linear_layer(v, linear.weight, linear.bias, activation=activation)


torch.fx.wrap("linear_wrapper")


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
