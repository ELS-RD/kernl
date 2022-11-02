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

import torch

from kernl.implementations.layer_norm import _layer_norm_fwd_fused_single_pass, layer_norm
from kernl.utils.extended_matcher import replace_pattern


def layer_norm_wrapper(v: torch.Tensor, layernorm: torch.nn.LayerNorm):
    # small hack to avoid casting weights/bias at each call
    if layernorm.weight.dtype == torch.float32:
        layernorm.weight.data = layernorm.weight.data.half()
    if layernorm.bias.dtype == torch.float32:
        layernorm.bias.data = layernorm.bias.data.half()

    return layer_norm(v, layernorm.weight, layernorm.bias, layernorm.eps)


def layer_norm_rms_wrapper(v: torch.Tensor, weight: torch.Tensor, eps: float):
    return layer_norm(v, weight, None, eps, _layer_norm_fwd_fused_single_pass, use_rms_norm=True)


torch.fx.wrap("layer_norm_wrapper")
torch.fx.wrap("layer_norm_rms_wrapper")


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


def replace_layer_norm_rms(gm: torch.fx.GraphModule):
    def pattern(v, weight):
        to_38 = v.to(torch.float32)
        pow_32 = to_38.pow(2)
        mean_31 = pow_32.mean(-1, keepdim=True)
        add_68 = mean_31 + 1e-06
        rsqrt_31 = torch.rsqrt(add_68)
        mul_69 = v * rsqrt_31
        mul_70 = weight * mul_69
        return mul_70

    def replace(v, weight):
        return layer_norm_rms_wrapper(v, weight, 1e-06)

    replace_pattern(gm, pattern, replace)
