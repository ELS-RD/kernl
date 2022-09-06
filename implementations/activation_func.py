import math

import triton
import triton.language as tl

# CREDITS: Initially inspired by the xformers repository

sqrt2pi = math.sqrt(2.0/math.pi)
sqrt2 = math.sqrt(2.0)


@triton.jit
def tanh(x):
    return tl.libdevice.tanh(x)


@triton.jit
def gelu(x):
    return x * 0.5 * (1.0 + tl.libdevice.erf(x / sqrt2))


@triton.jit
def fast_gelu(x):
    return 0.5 * x * (1 + tanh(sqrt2pi * (x + 0.044715 * x * x * x)))
