import torch

from implementations.layer_norm import layer_norm_forward
from utils.extended_matcher import replace_pattern

torch.fx.wrap('layer_norm_forward')


def replace_layer_norm(gm: torch.fx.GraphModule):
    class Pattern(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layernorm = torch.nn.LayerNorm((1, 1))

        def forward(self, v):
            return self.layernorm(v)

    class Replacement(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layernorm = torch.nn.LayerNorm((1, 1))

        def forward(self, v):
            return layer_norm_forward(v, self.layernorm.normalized_shape, self.layernorm.weight, self.layernorm.bias,
                                      self.layernorm.eps)

    replace_pattern(gm, Pattern(), Replacement())
