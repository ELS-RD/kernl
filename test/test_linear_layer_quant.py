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

import pytest
import torch

from conftest import assert_all_close, set_seed

from kernl.implementations.linear_layer import linear_layer
from kernl.implementations.linear_layer_quant import W8A8B8O8Linear, W8A8B8O8LinearReLU, W8A8BFP32OFP32Linear


@set_seed()
@pytest.mark.parametrize("implementation", ["triton", "pytorch"])
def test_quant_linear_a8_w8_b32_o32(benchmark, implementation):
    alpha, beta = 0.01, 0.0001
    B, M, N = 128, 512, 1024
    min_int8 = torch.iinfo(torch.int8).min
    max_int8 = torch.iinfo(torch.int8).max
    min_bias = int(torch.finfo(torch.float16).min)  # because bias will be converted to fp16 for benchmark reason
    max_bias = int(torch.finfo(torch.float16).max)
    weight = torch.randint(min_int8, max_int8, (N, M), dtype=torch.int8, device="cuda")
    bias = torch.randint(min_bias, max_bias, (N,), dtype=torch.int32, device="cuda")
    x = torch.randint(min_int8, max_int8, (B, M), dtype=torch.int8, device="cuda")
    linear = torch.nn.Linear(M, N, bias=True)
    linear.weight.data = weight.half() * alpha
    linear.bias.data = bias.half() * beta
    y_pytorch = linear(x.half())
    assert torch.all(torch.isfinite(y_pytorch))

    if implementation == "triton":
        y_triton = torch.zeros((B, N), device="cuda")
        y_triton = linear_layer(
            x=x,
            weight=weight,
            bias=bias,
            activation="",
            act_inputs=None,
            alpha_scaler=alpha,
            beta_scaler=beta,
            output=y_triton,
        )
        assert_all_close(y_pytorch, y_triton.half(), rtol=0, atol=4)  # not eq as baseline is computed with floats
        fn = lambda: linear_layer(  # noqa: E731
            x=x,
            weight=weight,
            bias=bias,
            activation="",
            act_inputs=None,
            alpha_scaler=alpha,
            beta_scaler=beta,
            output=y_triton,
        )
    elif implementation == "pytorch":
        x = x.half()
        fn = lambda: linear(x)  # noqa: E731
    else:
        raise ValueError(f"Unknown implementation: {implementation}")

    benchmark(fn)


@set_seed()
@torch.no_grad()
@pytest.mark.parametrize("implementation", ["w8a8b8o8", "w8a8b8o8_relu", "w8a8bfp32ofp32"])
def test_w8a8b8o8_linear_relu(implementation):
    B, M, N = 128, 512, 1024
    x = torch.randn(B, M)
    x_scale = x.abs().max() / 127
    qx = (x / x_scale).round().to(torch.int8)
    linear = torch.nn.Linear(M, N, bias=True)
    y_pytorch = linear(x)
    if implementation == "w8a8b8o8_relu":
        y_pytorch = y_pytorch.clamp(min=0)
    if implementation != "w8a8bfp32ofp32":
        y_scale = y_pytorch.abs().max() / 127
    # linear_quant = W8A8B8O8LinearReLU.from_float(linear, x_scale, y_scale).cuda()
    if implementation == "w8a8b8o8":
        linear_quant = W8A8B8O8Linear.from_float(linear, x_scale, y_scale).cuda()
    elif implementation == "w8a8b8o8_relu":
        linear_quant = W8A8B8O8LinearReLU.from_float(linear, x_scale, y_scale).cuda()
    elif implementation == "w8a8bfp32ofp32":
        linear_quant = W8A8BFP32OFP32Linear.from_float(linear, x_scale).cuda()
    else:
        raise ValueError(f"Unknown implementation: {implementation}")
    q_y = linear_quant(qx.cuda()).cpu()
    y_quant = q_y * y_scale if implementation != "w8a8bfp32ofp32" else q_y
    assert_all_close(y_pytorch, y_quant, rtol=0, atol=1e-1)
