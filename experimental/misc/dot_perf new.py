import torch
import triton
import triton.language as tl


torch.manual_seed(123)


@triton.jit
def kernel(
    Vec,
    Matrix,
    Out,
    vec_n_head_stride,
    vec_seqlen_stride,
    vec_d_head_stride,
    matrix_n_head_stride,
    matrix_seqlen_stride,
    matrix_d_head_stride,
    out_n_head_stride,
    out_seqlen_stride,
    out_d_head_stride,
    SIZE_N: tl.constexpr,
    D_HEAD: tl.constexpr,
    ILP: tl.constexpr,
):
    n_block_idx = tl.program_id(0)
    n_head_idx = tl.program_id(1)
    size_n_arange = tl.arange(0, SIZE_N)
    d_head_arange = tl.arange(0, D_HEAD)

    vec_ptr = Vec + n_head_idx * vec_n_head_stride + vec_d_head_stride * d_head_arange[None, :]
    vec = tl.load(vec_ptr)

    for start in range(ILP):
        # transpose matrix
        matrix_ptr = (
            Matrix
            + n_head_idx * matrix_n_head_stride
            + d_head_arange[None, :] * matrix_d_head_stride
            + ((start + n_block_idx * ILP) * SIZE_N + size_n_arange)[:, None] * matrix_seqlen_stride
        )
        mat = tl.load(matrix_ptr)
        result = mat.to(tl.float32) * vec.to(tl.float32)
        result = tl.sum(result, axis=1) + start * 0
        out_ptr = (
            Out
            + n_head_idx * out_n_head_stride
            + size_n_arange * out_d_head_stride
            + (start + n_block_idx * ILP) * SIZE_N
        )
        tl.store(out_ptr, result)


def triton_wrapper(vec: torch.Tensor, matrix: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
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


def reference_pytorch(vec: torch.Tensor, matrix: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
    return torch.matmul(input=vec, other=matrix.transpose(-1, -2), out=output)


size_n = 16
seq_len_k = 1024
n_head = 20
d_head = 64

q = torch.randn((n_head, 1, d_head), dtype=torch.float16, device="cuda")
k = torch.randn((n_head, seq_len_k, d_head), dtype=torch.float16, device="cuda")
out = torch.zeros((n_head, 1, seq_len_k), dtype=torch.float16, device="cuda")
expected = torch.zeros_like(out)

n_repeat = 10000
ILP = 2
grid = (triton.cdiv(seq_len_k, size_n * ILP), n_head)
print(grid)
print("CUDA times")

start_event = [torch.cuda.Event(enable_timing=True) for _ in range(n_repeat)]
end_event = [torch.cuda.Event(enable_timing=True) for _ in range(n_repeat)]
# warmup
for _ in range(n_repeat):
    triton_wrapper(vec=q, matrix=k, output=out)
    reference_pytorch(vec=q, matrix=k, output=expected)
torch.cuda.synchronize()
for i in range(n_repeat):
    start_event[i].record()
    reference_pytorch(vec=q, matrix=k, output=expected)
    end_event[i].record()
    torch.cuda.synchronize()
times_pytorch = torch.median(torch.tensor([s.elapsed_time(e) for s, e in zip(start_event, end_event)]))
# run
torch.cuda.synchronize()
for i in range(n_repeat):
    start_event[i].record()
    triton_wrapper(vec=q, matrix=k, output=out)
    end_event[i].record()
    torch.cuda.synchronize()
times_triton = torch.median(torch.tensor([s.elapsed_time(e) for s, e in zip(start_event, end_event)]))

reference_pytorch(vec=q, matrix=k, output=expected)
assert torch.allclose(out, expected, atol=1e-2), f"{out}\n{expected}"
print(f"{'tl.sum(a * b, 1)':<20}{times_triton.item() :.9f}")
print(f"{'vec @ matrix.t()':<20}{times_pytorch.item() :.9f}")
