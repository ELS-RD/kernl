import torch
import triton
import triton.language as tl


torch.manual_seed(5)

configs = []
for block_size_n in [16, 32, 64, 128, 256]:
    for ilp in [1, 2, 4, 8]:
        for num_warps in [1, 2, 4, 8]:
            for num_stages in [1, 2, 4, 8]:
                configs.append(
                    triton.Config({"SIZE_N": block_size_n, "ILP": ilp}, num_warps=num_warps, num_stages=num_stages)
                )


@triton.autotune(
    configs=[
        triton.Config({"SIZE_N": 16}, num_warps=1, num_stages=1),
        triton.Config({"SIZE_N": 8}, num_warps=1, num_stages=1),
    ],
    key=["matrix_rows_stride", "matrix_cols_stride"],
)
@triton.jit
def kernel(
    Vec,
    Matrix,
    Out,
    vec_batch_stride,
    vec_n_head_stride,
    vec_rows_stride,
    vec_cols_stride,
    matrix_batch_stride,
    matrix_n_head_stride,
    matrix_rows_stride,
    matrix_cols_stride,
    out_batch_stride,
    out_n_head_stride,
    out_rows_stride,
    out_cols_stride,
    VEC_N_COLS: tl.constexpr,
    VEC_SOFTMAX: tl.constexpr,
    SIZE_N: tl.constexpr,
):
    n_block_idx = tl.program_id(0)
    batch_idx = tl.program_id(1)
    n_head_idx = tl.program_id(2)

    size_n_arange = tl.arange(0, SIZE_N)
    vec_n_cols = tl.arange(0, VEC_N_COLS)

    vec_ptr = (
        Vec + batch_idx * vec_batch_stride + n_head_idx * vec_n_head_stride + vec_cols_stride * vec_n_cols[None, :]
    )

    vec = tl.load(vec_ptr, mask=vec_n_cols[None, :] < VEC_N_COLS, other=0.0).to(tl.float32)
    if VEC_SOFTMAX:
        vec_max = tl.max(vec, axis=1)
        vec = vec - vec_max[:, None]
        vec = tl.exp(vec)
        vec = vec / tl.sum(vec, axis=1)[:, None]
    matrix_ptr = (
        Matrix
        + batch_idx * matrix_batch_stride
        + n_head_idx * matrix_n_head_stride
        + vec_n_cols[None, :] * matrix_rows_stride
        + (n_block_idx * SIZE_N + size_n_arange)[:, None] * matrix_cols_stride
    )
    matrix_mask = (vec_n_cols[None, :] < VEC_N_COLS) & (size_n_arange[:, None] < SIZE_N)
    mat = tl.load(matrix_ptr, mask=matrix_mask, other=0.0).to(tl.float32)
    result = vec * mat
    result = tl.sum(result, axis=1)
    out_ptr = (
        Out
        + batch_idx * out_batch_stride
        + n_head_idx * out_n_head_stride
        + (n_block_idx * SIZE_N + size_n_arange) * out_cols_stride
    )
    tl.store(out_ptr, result, eviction_policy="evict_first")


def triton_wrapper(vec: torch.Tensor, matrix: torch.Tensor, output: torch.Tensor, transpose_mat: bool, softmax_vec: bool) -> torch.Tensor:
    matrix_stride = list(matrix.stride())
    if transpose_mat:
        # to transpose matrix we just need to swap strides!
        matrix_stride[-1], matrix_stride[-2] = matrix_stride[-2], matrix_stride[-1]
    mat_seq_len = matrix.shape[2 if transpose_mat else 3]
    vec_n_cols = vec.shape[3]
    assert vec.shape[2] == output.shape[2] == 1
    assert mat_seq_len == output.shape[3]
    assert vec.shape[3] == matrix.shape[3 if transpose_mat else 2]
    grid = lambda args: (
        triton.cdiv(mat_seq_len, args["SIZE_N"]),
        batch_size,
        n_head,
    )
    kernel[grid](
        vec,
        matrix,
        output,
        *vec.stride(),
        *matrix_stride,
        *output.stride(),
        vec_n_cols,
        softmax_vec,
    )
    return output


def reference_pytorch(
    vec: torch.Tensor, matrix: torch.Tensor, output: torch.Tensor, transpose_mat: bool, softmax_vec: bool
) -> torch.Tensor:
    if transpose_mat:
        matrix = matrix.transpose(-1, -2)
    if softmax_vec:
        vec = torch.nn.functional.softmax(vec, dim=-1, dtype=torch.float32).half()
    return torch.matmul(input=vec, other=matrix, out=output)


batch_size = 5
n_head = 20
seq_len_k = 1024
d_head = 64


cache = torch.empty(int(256e6), dtype=torch.int8, device="cuda")
q = torch.randn((batch_size, n_head, 1, d_head), dtype=torch.float16, device="cuda")
k = torch.randn((batch_size, n_head, seq_len_k, d_head), dtype=torch.float16, device="cuda")
out_qkt = torch.zeros((batch_size, n_head, 1, seq_len_k), dtype=torch.float16, device="cuda")
expected_qkt = torch.zeros_like(out_qkt)

qkt = torch.randn((batch_size, n_head, 1, seq_len_k), dtype=torch.float16, device="cuda")

v = torch.randn((batch_size, n_head, seq_len_k, d_head), dtype=torch.float16, device="cuda").permute(0, 1, 3, 2).contiguous().permute(0, 1, 3, 2)
out_qktv = torch.zeros((batch_size, n_head, 1, d_head), dtype=torch.float16, device="cuda")
expected_qktv = torch.zeros_like(out_qktv)

print("CUDA times")
n_repeat = 1000
start_event = [torch.cuda.Event(enable_timing=True) for _ in range(n_repeat)]
end_event = [torch.cuda.Event(enable_timing=True) for _ in range(n_repeat)]

# warmup
for _ in range(n_repeat):
    triton_wrapper(vec=q, matrix=k, output=out_qkt, transpose_mat=True, softmax_vec=False)
    reference_pytorch(vec=q, matrix=k, output=expected_qkt, transpose_mat=True, softmax_vec=False)
    triton_wrapper(vec=qkt, matrix=v, output=out_qktv, transpose_mat=False, softmax_vec=True)
    reference_pytorch(vec=qkt, matrix=v, output=expected_qktv, transpose_mat=False, softmax_vec=True)
torch.cuda.synchronize()

# run PyTorch QK^t
for i in range(n_repeat):
    torch.cuda.synchronize()
    start_event[i].record()
    reference_pytorch(vec=q, matrix=k, output=expected_qkt, transpose_mat=True, softmax_vec=False)
    end_event[i].record()
    cache.zero_()

times_pytorch_qkt = torch.median(torch.tensor([s.elapsed_time(e) for s, e in zip(start_event, end_event)]))

# run Triton QK^t
for i in range(n_repeat):
    torch.cuda.synchronize()
    start_event[i].record()
    triton_wrapper(vec=q, matrix=k, output=out_qkt, transpose_mat=True, softmax_vec=False)
    end_event[i].record()
    cache.zero_()

times_triton_t_qkt = torch.median(torch.tensor([s.elapsed_time(e) for s, e in zip(start_event, end_event)]))

# run PyTorch QK^tV
for i in range(n_repeat):
    torch.cuda.synchronize()
    start_event[i].record()
    reference_pytorch(vec=qkt, matrix=v, output=expected_qktv, transpose_mat=False, softmax_vec=True)
    end_event[i].record()
    cache.zero_()

times_pytorch_qktv = torch.median(torch.tensor([s.elapsed_time(e) for s, e in zip(start_event, end_event)]))

# run Triton QK^tV
for i in range(n_repeat):
    torch.cuda.synchronize()
    start_event[i].record()
    triton_wrapper(vec=qkt, matrix=v, output=out_qktv, transpose_mat=False, softmax_vec=True)
    end_event[i].record()
    cache.zero_()

times_triton_t_qktv = torch.median(torch.tensor([s.elapsed_time(e) for s, e in zip(start_event, end_event)]))


assert torch.allclose(out_qkt, expected_qkt, atol=1e-1), f"{out_qkt}\n{expected_qkt}"
assert torch.allclose(out_qktv, expected_qktv, atol=1e-1), f"{out_qktv}\n{expected_qktv}"
print(f"{'tl.sum(q * k^t, 1)':<20}{times_triton_t_qkt.item() :.9f}")
print(f"{'q @ k.t()':<20}{times_pytorch_qkt.item() :.9f}")
print(f"{'tl.sum(qkt * v, 1)':<20}{times_triton_t_qktv.item() :.9f}")
print(f"{'qkt @ v':<20}{times_pytorch_qktv.item() :.9f}")

# note: for the per-transposed, if size n >= n head, timings are similar to not transposed version.
