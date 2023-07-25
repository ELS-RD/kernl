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
# code inspired from torch-int pacakge
# https://github.com/Guangxuan-Xiao/torch-int/blob/main/torch_int/nn/linear.py

import torch

from kernl.implementations.linear_layer import linear_layer


@torch.no_grad()
def quantize_per_tensor_absmax(t: torch.Tensor):
    scale = t.abs().max() / 127
    if not t.is_cuda:
        # half rounding is not supported on CPU
        t = t.float()
    # use inplace operation to save memory
    t.div_(scale).round_()
    t_q = t.to(torch.int8)
    return t_q, scale


class W8A8B8O8Linear(torch.nn.Module):
    # For qkv_proj
    def __init__(self, in_features, out_features, alpha=1.0, beta=1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.register_buffer(
            "weight",
            torch.randint(-127, 127, (self.out_features, self.in_features), dtype=torch.int8, requires_grad=False),
        )
        self.register_buffer("bias", torch.zeros((1, self.out_features), dtype=torch.int8, requires_grad=False))
        self.register_buffer("a", torch.tensor(alpha))
        self.register_buffer("b", torch.tensor(beta))

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.weight = self.weight.to(*args, **kwargs)
        self.bias = self.bias.to(*args, **kwargs)
        return self

    @torch.no_grad()
    def forward(self, x):
        x_shape = x.shape
        x = x.view(-1, x_shape[-1])
        y = torch.empty((x.shape[0], self.weight.shape[0]), device=x.device, dtype=torch.int8)
        linear_layer(
            x=x,
            weight=self.weight,
            bias=self.bias,
            activation="",
            act_inputs=None,
            alpha_scaler=self.a.item(),
            beta_scaler=self.b.item(),
            output=y,
        )
        y = y.view(*x_shape[:-1], -1)
        return y

    @staticmethod
    def from_float(module: torch.nn.Linear, input_scale, output_scale):
        int8_module = W8A8B8O8Linear(module.in_features, module.out_features)
        int8_weight, weight_scale = quantize_per_tensor_absmax(module.weight)
        int8_bias, bias_scale = quantize_per_tensor_absmax(module.bias)
        alpha = input_scale * weight_scale / output_scale
        beta = bias_scale / output_scale
        int8_module.weight = int8_weight
        int8_module.bias = int8_bias
        int8_module.a = alpha
        int8_module.b = beta
        return int8_module


class W8A8B8O8LinearReLU(torch.nn.Module):
    # For fc1
    def __init__(self, in_features, out_features, alpha=1.0, beta=1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.register_buffer(
            "weight",
            torch.randint(-127, 127, (self.out_features, self.in_features), dtype=torch.int8, requires_grad=False),
        )
        self.register_buffer("bias", torch.zeros((1, self.out_features), dtype=torch.int8, requires_grad=False))
        self.register_buffer("a", torch.tensor(alpha))
        self.register_buffer("b", torch.tensor(beta))

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.weight = self.weight.to(*args, **kwargs)
        self.bias = self.bias.to(*args, **kwargs)
        return self

    @torch.no_grad()
    def forward(self, x):
        x_shape = x.shape
        x = x.view(-1, x_shape[-1])
        y = torch.empty((x.shape[0], self.weight.shape[0]), device=x.device, dtype=torch.int8)

        linear_layer(
            x=x,
            weight=self.weight,
            bias=self.bias,
            activation="relu",
            act_inputs=None,
            alpha_scaler=self.a.item(),
            beta_scaler=self.b.item(),
            output=y,
        )
        y = y.view(*x_shape[:-1], -1)
        return y

    @staticmethod
    def from_float(module: torch.nn.Linear, input_scale, output_scale):
        # TODO: add zero-point to prevent the bit waste
        int8_module = W8A8B8O8LinearReLU(module.in_features, module.out_features)
        int8_weight, weight_scale = quantize_per_tensor_absmax(module.weight)
        int8_bias, bias_scale = quantize_per_tensor_absmax(module.bias)
        alpha = input_scale * weight_scale / output_scale
        beta = bias_scale / output_scale
        int8_module.weight = int8_weight
        int8_module.bias = int8_bias
        int8_module.a = alpha
        int8_module.b = beta
        return int8_module


class W8A8BFP32OFP32Linear(torch.nn.Module):
    # For fc2 and out_proj
    def __init__(self, in_features, out_features, alpha=1.0, beta=1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.register_buffer(
            "weight",
            torch.randint(-127, 127, (self.out_features, self.in_features), dtype=torch.int8, requires_grad=False),
        )
        self.register_buffer("bias", torch.zeros((1, self.out_features), dtype=torch.float32, requires_grad=False))
        self.register_buffer("a", torch.tensor(alpha))

    def _apply(self, fn):
        # prevent the bias from being converted to half
        super()._apply(fn)
        self.bias = self.bias.to(torch.float32)
        return self

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.weight = self.weight.to(*args, **kwargs)
        self.bias = self.bias.to(*args, **kwargs)
        self.bias = self.bias.to(torch.float32)
        return self

    @torch.no_grad()
    def forward(self, x):
        x_shape = x.shape
        x = x.view(-1, x_shape[-1])
        self.bias = self.bias.to(torch.float32)
        y = torch.empty((x.shape[0], self.weight.shape[0]), device=x.device, dtype=torch.float32)

        linear_layer(
            x=x,
            weight=self.weight,
            bias=self.bias,
            activation="",
            act_inputs=None,
            alpha_scaler=self.a.item(),
            beta_scaler=1.0,
            output=y,
        )
        y = y.view(*x_shape[:-1], -1)
        return y

    @staticmethod
    def from_float(module: torch.nn.Linear, input_scale):
        int8_module = W8A8BFP32OFP32Linear(module.in_features, module.out_features)
        int8_weight, weight_scale = quantize_per_tensor_absmax(module.weight)
        alpha = input_scale * weight_scale
        int8_module.weight = int8_weight
        int8_module.bias = module.bias.to(torch.float32)
        int8_module.a = alpha
        int8_module.input_scale = input_scale
        int8_module.weight_scale = weight_scale
        return int8_module
