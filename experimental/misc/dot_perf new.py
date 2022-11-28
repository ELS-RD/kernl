import torch
import triton
import triton.language as tl


torch.manual_seed(123)


@triton.jit
def kernel(
    Vec,
    Matrix,
    Out,
    vec_seqlen_stride,
    vec_d_head_stride,
    matrix_seqlen_stride,
    matrix_d_head_stride,
    out_seqlen_stride,
    out_d_head_stride,
    SIZE_N: tl.constexpr,
    D_HEAD: tl.constexpr,
    ILP: tl.constexpr,
):
    n_block_idx = tl.program_id(0)
    size_n_arange = tl.arange(0, SIZE_N)
    d_head_arange = tl.arange(0, D_HEAD)

    vec_ptr = Vec + vec_d_head_stride * d_head_arange[None, :]
    vec = tl.load(vec_ptr)

    for start in range(ILP):
        # transpose matrix
        matrix_ptr = Matrix + d_head_arange[None, :] * matrix_d_head_stride + ((start + n_block_idx * ILP) * SIZE_N + size_n_arange)[:, None] * matrix_seqlen_stride
        mat = tl.load(matrix_ptr)
        result = mat.to(tl.float32) * vec.to(tl.float32)
        result = tl.sum(result, axis=1) + start * 0
        out_ptr = Out + size_n_arange * out_d_head_stride + (start + n_block_idx * ILP) * SIZE_N
        tl.store(out_ptr, result)


size_n = 16
seq_len_k = 1024
# n_head = 20
d_head = 64

q = torch.randn((1, d_head), dtype=torch.float16, device="cuda")
k = torch.randn((seq_len_k, d_head), dtype=torch.float16, device="cuda")
out = torch.zeros((1, seq_len_k), dtype=torch.float16, device="cuda")

n_repeat = 10000
ILP = 2
grid = (triton.cdiv(seq_len_k, size_n * ILP), )
print(grid)
print("CUDA times")

start_event = [torch.cuda.Event(enable_timing=True) for _ in range(n_repeat)]
end_event = [torch.cuda.Event(enable_timing=True) for _ in range(n_repeat)]
# warmup
for _ in range(n_repeat):
    kernel[grid](
        q,
        k,
        out,
        *q.stride(),
        *k.stride(),
        *out.stride(),
        size_n,
        d_head,
        ILP,
    )
    q @ k.t()
torch.cuda.synchronize()
for i in range(n_repeat):
    start_event[i].record()
    q @ k.t()
    end_event[i].record()
    torch.cuda.synchronize()
times_pytorch = torch.median(torch.tensor([s.elapsed_time(e) for s, e in zip(start_event, end_event)]))
# run
torch.cuda.synchronize()
for i in range(n_repeat):
    start_event[i].record()
    kernel[grid](
        q,
        k,
        out,
        *q.stride(),
        *k.stride(),
        *out.stride(),
        size_n,
        d_head,
        ILP,
    )
    end_event[i].record()
    torch.cuda.synchronize()
times_triton = torch.median(torch.tensor([s.elapsed_time(e) for s, e in zip(start_event, end_event)]))

expected = q @ k.t()
assert torch.allclose(out, expected, atol=1e-4), f"{out}\n{expected}"
print(f"{'tl.sum(a * b, 1)':<20}{times_triton.item() :.9f}")
print(f"{'vec @ matrix.t()':<20}{times_pytorch.item() :.9f}")
