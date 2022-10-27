#  Copyright 2022 Lefebvre Sarrut
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

from typing import Callable

import torch

from kernl.implementations.linear_layer import linear_layer
from kernl.utils.extended_matcher import replace_pattern


def linear_wrapper(v: torch.Tensor, linear: torch.nn.Linear, activation=""):
    return linear_wrapper_functional(v, linear.weight, linear.bias, activation=activation)


torch.fx.wrap("linear_wrapper")


def linear_wrapper_functional(v: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, activation=""):
    # small hack to avoid casting weights/bias at each call
    if weight.dtype == torch.float32:
        weight.data = weight.data.half()
    if bias is not None and bias.dtype == torch.float32:
        bias.data = bias.data.half()

    return linear_layer(v, weight, bias, activation=activation)


torch.fx.wrap("linear_wrapper_functional")


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


def replace_linear_fn(gm: torch.fx.GraphModule):
    def pattern(v, weight, bias):
        return torch.nn.functional.linear(v, weight, bias)

    def replace(v, weight, bias):
        output = linear_wrapper_functional(v, weight, bias)
        return output

    replace_pattern(gm, pattern, replace)


# Todo: to be removed when we support dynamic constant match
def replace_linear_fn_constant_match(gm: torch.fx.GraphModule):
    def pattern(v, weight):
        return torch.nn.functional.linear(v, weight, None)

    def replace(v, weight):
        output = linear_wrapper_functional(v, weight, None)
        return output

    replace_pattern(gm, pattern, replace)


def replace_all_linear(gm: torch.fx.GraphModule):
    replace_linear_activation(gm, torch.nn.Tanh(), "tanh")
    replace_linear_activation(gm, torch.nn.ReLU(), "relu")
    replace_linear_activation(gm, torch.nn.functional.gelu, "gelu")
    replace_linear(gm)
    replace_linear_fn(gm)
    replace_linear_fn_constant_match(gm)
