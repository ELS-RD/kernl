import math

import triton
import triton.language as tl


@triton.jit
def tanh(x):
    return tl.libdevice.tanh(x)

sqrt2pi = math.sqrt(2.0/math.pi)
@triton.jit
def gelu(x):
    return 0.5 * x * (1 + tanh(sqrt2pi * (x + 0.044715 * x * x * x)))