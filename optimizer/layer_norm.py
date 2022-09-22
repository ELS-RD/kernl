import torch

from implementations.layer_norm import layer_norm
from utils.extended_matcher import replace_pattern


def layer_norm_wrapper(v, layernorm):
    # small hack to avoid casting weights/bias at each call
    if not hasattr(layernorm, "weight_h"):
        layernorm.weight_h = layernorm.weight.half()
        layernorm.weight = None
    if not hasattr(layernorm, "bias_h"):
        layernorm.bias_h = layernorm.bias.half()
        layernorm.bias = None

    return layer_norm(v, layernorm.weight_h, layernorm.bias_h, layernorm.eps)


torch.fx.wrap('layer_norm_wrapper')


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
            return layer_norm_wrapper(v, self.layernorm)

    replace_pattern(gm, Pattern(), Replacement())
