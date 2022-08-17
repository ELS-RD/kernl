import triton
import triton.language as tl


@triton.jit
def tanh(x):
    return tl.libdevice.tanh(x)
