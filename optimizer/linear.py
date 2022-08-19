import torch

from implementations.linear_layer import linear_layer
from utils.extended_matcher import replace_pattern

torch.fx.wrap('linear_layer')

def replace_linear_activation(gm: torch.fx.GraphModule):
    class Pattern(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(1, 1)
            self.activation = torch.nn.Tanh()

        def forward(self, v):
            return self.activation(self.linear(v))

    class Replacement(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(1, 1)
            self.activation = torch.nn.Tanh()

        def forward(self, v):
            linear = linear_layer(v, self.linear.weight.data, self.linear.bias.data, activation="tanh")
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
            output, _ = linear_layer(v, self.linear.weight.data, self.linear.bias.data)
            return output

    replace_pattern(gm, Pattern(), Replacement())


def replace_all_linear(gm: torch.fx.GraphModule):
    replace_linear_activation(gm)
    replace_linear(gm)
