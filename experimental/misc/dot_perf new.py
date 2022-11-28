import torch
import triton
import triton.language as tl


torch.manual_seed(123)

configs = []
for block_size_n in [16, 32, 64, 128, 256]:
    for ilp in [1, 2, 4, 8]:
        for num_warps in [1, 2, 4, 8]:
            for num_stages in [1, 2, 4, 8]:
                configs.append(
                    triton.Config({"SIZE_N": block_size_n, "ILP": ilp}, num_warps=num_warps, num_stages=num_stages)
                )


@triton.autotune(
    configs=[triton.Config({"SIZE_N": 16, "ILP": 2}, num_warps=1, num_stages=2)],
    key=[],
)
@triton.jit
def kernel(
    Vec,
    Matrix,
    Out,
    vec_batch_stride,
    vec_n_head_stride,
    vec_seqlen_stride,
    vec_d_head_stride,
    matrix_batch_stride,
    matrix_n_head_stride,
    matrix_seqlen_stride,
    matrix_d_head_stride,
    out_batch_stride,
    out_n_head_stride,
    out_seqlen_stride,
    out_d_head_stride,
    D_HEAD: tl.constexpr,
    TRANSPOSE_MAT: tl.constexpr,
    ILP: tl.constexpr,
    SIZE_N: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    n_head_idx = tl.program_id(1)
    n_block_idx = tl.program_id(2)

    size_n_arange = tl.arange(0, SIZE_N)
    d_head_arange = tl.arange(0, D_HEAD)

    vec_ptr = (
        Vec + batch_idx * vec_batch_stride + n_head_idx * vec_n_head_stride + vec_d_head_stride * d_head_arange[None, :]
    )
    vec = tl.load(vec_ptr).to(tl.float32)

    for start in range(ILP):
        # transpose matrix
        if TRANSPOSE_MAT:
            matrix_ptr = (
                Matrix
                + batch_idx * matrix_batch_stride
                + n_head_idx * matrix_n_head_stride
                + d_head_arange[None, :] * matrix_d_head_stride
                + ((start + n_block_idx * ILP) * SIZE_N + size_n_arange)[:, None] * matrix_seqlen_stride
            )
        else:
            matrix_ptr = (
                Matrix
                + batch_idx * matrix_batch_stride
                + n_head_idx * matrix_n_head_stride
                + d_head_arange[None, :] * matrix_seqlen_stride
                + ((start + n_block_idx * ILP) * SIZE_N + size_n_arange)[:, None] * matrix_d_head_stride
            )

        mat = tl.load(matrix_ptr).to(tl.float32)
        result = mat * vec
        result = tl.sum(result, axis=1)
        out_ptr = (
            Out
            + batch_idx * out_batch_stride
            + n_head_idx * out_n_head_stride
            + size_n_arange * out_d_head_stride
            + (start + n_block_idx * ILP) * SIZE_N
        )
        tl.store(out_ptr, result)


def triton_wrapper(vec: torch.Tensor, matrix: torch.Tensor, output: torch.Tensor, transpose_mat: bool) -> torch.Tensor:
    assert vec.shape[2] == output.shape[2] == 1
    # assert matrix.shape[3] == output.shape[3]
    assert vec.shape[3] == matrix.shape[3 if transpose_mat else 2]

    grid = lambda args: (batch_size, n_head, triton.cdiv(seq_len_k, args["SIZE_N"] * args["ILP"]))
    kernel[grid](
        vec,
        matrix,
        output,
        *vec.stride(),
        *matrix.stride(),
        *output.stride(),
        d_head,
        transpose_mat,
    )
    return output


def reference_pytorch(
    vec: torch.Tensor, matrix: torch.Tensor, output: torch.Tensor, transpose_mat: bool
) -> torch.Tensor:
    if transpose_mat:
        matrix = matrix.transpose(-1, -2)
    return torch.matmul(input=vec, other=matrix, out=output)


batch_size = 1
n_head = 20
seq_len_k = 1024
d_head = 64

q = torch.randn((batch_size, n_head, 1, d_head), dtype=torch.float16, device="cuda")
k = torch.randn((batch_size, n_head, seq_len_k, d_head), dtype=torch.float16, device="cuda")
k_t = k.transpose(-1, -2).contiguous()
out = torch.zeros((batch_size, n_head, 1, seq_len_k), dtype=torch.float16, device="cuda")
out_t = torch.zeros_like(out)
expected = torch.zeros_like(out)


# triton_wrapper(vec=q, matrix=k, output=out, transpose_mat=False)
# reference_pytorch(vec=q, matrix=k, output=expected, transpose_mat=False)
# print(out)
# print(expected)

print("CUDA times")
n_repeat = 1000
start_event = [torch.cuda.Event(enable_timing=True) for _ in range(n_repeat)]
end_event = [torch.cuda.Event(enable_timing=True) for _ in range(n_repeat)]

# warmup
for _ in range(n_repeat):
    triton_wrapper(vec=q, matrix=k, output=out, transpose_mat=True)
    triton_wrapper(vec=q, matrix=k_t, output=out_t, transpose_mat=False)
    reference_pytorch(vec=q, matrix=k, output=expected, transpose_mat=True)
torch.cuda.synchronize()

# run PyTorch
for i in range(n_repeat):
    start_event[i].record()
    reference_pytorch(vec=q, matrix=k, output=expected, transpose_mat=True)
    end_event[i].record()
    torch.cuda.synchronize()
times_pytorch = torch.median(torch.tensor([s.elapsed_time(e) for s, e in zip(start_event, end_event)]))
torch.cuda.synchronize()

# run Triton transpose
for i in range(n_repeat):
    start_event[i].record()
    triton_wrapper(vec=q, matrix=k, output=out, transpose_mat=True)
    end_event[i].record()
    torch.cuda.synchronize()
times_triton_t = torch.median(torch.tensor([s.elapsed_time(e) for s, e in zip(start_event, end_event)]))

# run Triton pre-transpose
for i in range(n_repeat):
    start_event[i].record()
    triton_wrapper(vec=q, matrix=k_t, output=out, transpose_mat=False)
    end_event[i].record()
    torch.cuda.synchronize()
times_triton_pre_t = torch.median(torch.tensor([s.elapsed_time(e) for s, e in zip(start_event, end_event)]))

assert torch.allclose(out, expected, atol=1e-1), f"{out}\n{expected}"
assert torch.allclose(out, out_t, atol=1e-1), f"{out}\n{out_t}"
print(f"{'tl.sum(q * k^t, 1)':<20}{times_triton_t.item() :.9f}")
print(f"{'tl.sum(q * k, 1)':<20}{times_triton_pre_t.item() :.9f}")
print(f"{'q @ k.t()':<20}{times_pytorch.item() :.9f}")

# note: for the per-transposed, if size n >= n head, timings are similar to not transposed version.
