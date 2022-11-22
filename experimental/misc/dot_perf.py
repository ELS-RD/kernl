import torch
import triton
import triton.language as tl


torch.manual_seed(123)


@triton.jit
def kernel(
    V,
    M,
    O,
    vec_stride_x,
    matrix_stride_x,
    matrix_stride_y,
    out_stride_x,
    out_stride_y,
    SIZE_M: tl.constexpr,
    D_HEAD: tl.constexpr,
    IS_DOT: tl.constexpr,
):

    size_m_arange = tl.arange(0, SIZE_M)
    d_head_arange = tl.arange(0, D_HEAD)
    # transpose matrix
    matrix_ptr = M + d_head_arange[None, :] * matrix_stride_y + size_m_arange[:, None] * matrix_stride_x
    matrix = tl.load(matrix_ptr)
    out_ptr = O + size_m_arange * out_stride_y

    if IS_DOT:
        vec_ptr = V + vec_stride_x * size_m_arange[:, None] + vec_stride_x * d_head_arange[None, :]
        vec = tl.load(vec_ptr, mask=size_m_arange[:, None] < 1, other=0.0)
        result = tl.dot(matrix, vec, trans_a=False, trans_b=True)
    else:
        vec_ptr = V + vec_stride_x * d_head_arange[None, :]
        vec = tl.load(vec_ptr)
        result = matrix.to(tl.float32) * vec.to(tl.float32)

    result = tl.sum(result, axis=1)
    tl.store(out_ptr, result)


size_m = 16
d_head = 128

vec = torch.randn((d_head,), dtype=torch.float16, device="cuda")
matrix = torch.randn((size_m, d_head), dtype=torch.float16, device="cuda")
out = torch.zeros((1, size_m), dtype=torch.float16, device="cuda")

n_repeat = 10000
grid = (10000,)

for use_dot in [True, False]:
    print("-----------------------------------")
    if use_dot:
        print("dot")
    else:
        print("add,mul")

    start_event = [torch.cuda.Event(enable_timing=True) for _ in range(n_repeat)]
    end_event = [torch.cuda.Event(enable_timing=True) for _ in range(n_repeat)]
    torch.cuda.synchronize()
    for _ in range(n_repeat):
        kernel[grid](
            vec,
            matrix,
            out,
            *vec.stride(),
            *matrix.stride(),
            *out.stride(),
            size_m,
            d_head,
            use_dot,
        )
    torch.cuda.synchronize()
    for i in range(n_repeat):
        start_event[i].record()
        kernel[grid](
            vec,
            matrix,
            out,
            *vec.stride(),
            *matrix.stride(),
            *out.stride(),
            size_m,
            d_head,
            use_dot,
        )
        end_event[i].record()
    torch.cuda.synchronize()

    times = torch.tensor([s.elapsed_time(e) for s, e in zip(start_event, end_event)])
    print("CUDA time %.4f" % torch.median(times).item())

    expect = vec @ matrix.t()
    print(f"max error={(out - expect).abs().max().item()}")
