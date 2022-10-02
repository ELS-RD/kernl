import math

import triton
import triton.language as tl


# CREDITS: Initially inspired by the xformers repository

sqrt2pi = math.sqrt(2.0 / math.pi)
sqrt2 = math.sqrt(2.0)


@triton.jit
def tanh(x):
    """Tanh activation function"""
    return tl.libdevice.tanh(x)


@triton.jit
def relu(x):
    """Relu activation function"""
    return tl.maximum(0, x)


@triton.jit
def fast_gelu(x):
    """Fast approximation of the gelu function. May slightly decrease accuracy."""
    return 0.5 * x * (1 + tanh(sqrt2pi * (x + 0.044715 * x * x * x)))


@triton.jit
def gelu(x):
    """Gaussian Error Linear Unit (GELU)"""
    return x * 0.5 * (1.0 + tl.libdevice.erf(x / sqrt2))
