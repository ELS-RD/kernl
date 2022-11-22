import torch
import triton
import triton.language as tl


torch.manual_seed(123)


@triton.jit
def kernel(
    M,
    Out,
    matrix_stridex,
    matrix_stridey,
    out_stridex,
    out_stridey,
    SIZE_M: tl.constexpr,
    D_HEAD: tl.constexpr,
):
    size_m_arange = tl.arange(0, SIZE_M)
    d_head_arange = tl.arange(0, D_HEAD)
    # transpose
    matrix_ptr = M + d_head_arange[None, :] * matrix_stridey + size_m_arange[:, None] * matrix_stridex
    out_ptr = Out + d_head_arange[None, :] * out_stridex + size_m_arange[:, None] * out_stridey
    matrix = tl.load(matrix_ptr)
    tl.store(out_ptr, matrix)


size_m = 16
d_head = 32

matrix = torch.randn((size_m, d_head), dtype=torch.float16, device="cuda")
out = torch.zeros((d_head, size_m), dtype=torch.float16, device="cuda")

grid = (1,)
kernel[grid](
    matrix,
    out,
    *matrix.stride(),
    *out.stride(),
    size_m,
    d_head,
)

assert torch.allclose(matrix.t(), out)
