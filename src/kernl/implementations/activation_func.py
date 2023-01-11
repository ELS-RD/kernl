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

import math

import triton
import triton.language as tl


# CREDITS: Initially inspired by the xformers repository

sqrt2pi = math.sqrt(2.0 / math.pi)
sqrt2 = math.sqrt(2.0)
gaussian_pdf_normalization = 1.0 / math.sqrt(2 * math.pi)


@triton.jit
def tanh(x):
    """Tanh activation function"""
    return tl.libdevice.tanh(x)


@triton.jit
def tanh_grad(x):
    """Tanh derivative function"""
    return 1 - tl.libdevice.pow(tl.libdevice.tanh(x), 2)


@triton.jit
def relu(x):
    """Relu activation function"""
    return tl.maximum(0, x)


@triton.jit
def relu_grad(x):
    """Relu derivative function"""
    return tl.maximum(0, x)


@triton.jit
def fast_gelu(x):
    """Fast approximation of the gelu function. May slightly decrease accuracy."""
    return 0.5 * x * (1 + tanh(sqrt2pi * (x + 0.044715 * x * x * x)))


@triton.jit
def fast_gelu_grad(x):
    """Derivative of fast approximation of the gelu function."""
    # CREDITS: Fast implementation proposed in
    # https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/fused_bias_gelu.py#L30
    tanh_out = tanh(0.79788456 * x * (1 + 0.044715 * x * x))
    return 0.5 * x * ((1 - tanh_out * tanh_out) * (0.79788456 + 0.1070322243 * x * x)) + 0.5 * (1 + tanh_out)


@triton.jit
def gelu(x):
    """Gaussian Error Linear Unit (GELU)"""
    return x * 0.5 * (1.0 + tl.libdevice.erf(x / sqrt2))


@triton.jit
def gelu_grad(x):
    """Derivative of Gaussian Error Linear Unit (GELU)"""
    cdf = 0.5 * (1.0 + tl.libdevice.erf(x * sqrt2))
    pdf = tl.exp(-0.5 * x * x) * gaussian_pdf_normalization
    return cdf + x * pdf
