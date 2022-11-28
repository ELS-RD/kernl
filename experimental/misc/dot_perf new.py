import torch
import triton
import triton.language as tl


torch.manual_seed(123)


@triton.jit
def kernel(
    V,
    M,
    Out,
    vec_stride_x,
    vec_stride_y,
    matrix_stride_x,
    matrix_stride_y,
    out_stride_x,
    out_stride_y,
    SIZE_N: tl.constexpr,
    D_HEAD: tl.constexpr,
    ILP: tl.constexpr,
):
    n_block_idx = tl.program_id(0)
    size_n_arange = tl.arange(0, SIZE_N)
    d_head_arange = tl.arange(0, D_HEAD)

    vec_ptr = V + vec_stride_y * d_head_arange[None, :]
    vec = tl.load(vec_ptr)

    for start in range(ILP):
        # transpose matrix
        matrix_ptr = M + d_head_arange[None, :] * matrix_stride_y + ((start + n_block_idx * ILP) * SIZE_N + size_n_arange)[:, None] * matrix_stride_x
        matrix = tl.load(matrix_ptr)
        result = matrix.to(tl.float32) * vec.to(tl.float32)
        result = tl.sum(result, axis=1) + start * 0
        out_ptr = Out + size_n_arange * out_stride_y + (start + n_block_idx * ILP) * SIZE_N
        tl.store(out_ptr, result)


size_n = 16
seq_len_k = 1024
d_head = 64

vec = torch.randn((1, d_head), dtype=torch.float16, device="cuda")
matrix = torch.randn((seq_len_k, d_head), dtype=torch.float16, device="cuda")
out = torch.zeros((1, seq_len_k), dtype=torch.float16, device="cuda")

n_repeat = 10000
ILP = 10
grid = (triton.cdiv(seq_len_k, size_n * ILP), )
print(grid)
print("CUDA times")

start_event = [torch.cuda.Event(enable_timing=True) for _ in range(n_repeat)]
end_event = [torch.cuda.Event(enable_timing=True) for _ in range(n_repeat)]
# warmup
for _ in range(n_repeat):
    kernel[grid](
        vec,
        matrix,
        out,
        *vec.stride(),
        *matrix.stride(),
        *out.stride(),
        size_n,
        d_head,
        ILP,
    )
torch.cuda.synchronize()
for i in range(n_repeat):
    start_event[i].record()
    vec @ matrix.t()
    end_event[i].record()
    torch.cuda.synchronize()
times_pytorch = torch.median(torch.tensor([s.elapsed_time(e) for s, e in zip(start_event, end_event)]))
# run
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
        size_n,
        d_head,
        ILP,
    )
    end_event[i].record()
    torch.cuda.synchronize()
times_triton = torch.median(torch.tensor([s.elapsed_time(e) for s, e in zip(start_event, end_event)]))

expected = vec @ matrix.t()
assert torch.allclose(out, expected, atol=1e-4), f"{out}\n{expected}"
print(f"{'tl.sum(a * b, 1)':<20}{times_triton.item() :.9f}")
print(f"{'vec @ matrix.t()':<20}{times_pytorch.item() :.9f}")
