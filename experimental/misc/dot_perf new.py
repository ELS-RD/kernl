from typing import Tuple

import torch
import triton
import triton.language as tl

from kernl.implementations.cuda_graph import cuda_graphs_wrapper


torch.manual_seed(123)


@triton.autotune(
    configs=[
        triton.Config({"SIZE_N": 32}, num_warps=1, num_stages=1),
        triton.Config({"SIZE_N": 16}, num_warps=1, num_stages=1),
        triton.Config({"SIZE_N": 8}, num_warps=1, num_stages=1),
        triton.Config({"SIZE_N": 4}, num_warps=1, num_stages=1),
        triton.Config({"SIZE_N": 2}, num_warps=1, num_stages=1),
        triton.Config({"SIZE_N": 1}, num_warps=1, num_stages=1),
    ],
    key=["vec_cols", "mat_rows", "mat_cols", "out_cols"],
)
@triton.jit
def mul_vec_mat(
    vec_cols: tl.constexpr,
    mat_rows: tl.constexpr,
    mat_cols: tl.constexpr,
    out_cols: tl.constexpr,
    vec_tensor,
    matrix_tensor,
    out_tensor,
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
    SCALER: tl.constexpr,
    VEC_COLS: tl.constexpr,
    VEC_SOFTMAX: tl.constexpr,
    SIZE_N: tl.constexpr,
):
    n_block_idx = tl.program_id(0)
    batch_idx = tl.program_id(1)
    n_head_idx = tl.program_id(2)

    size_n_arange = tl.arange(0, SIZE_N)
    vec_arange = tl.arange(0, VEC_COLS)

    vec_ptr = (
        vec_tensor
        + batch_idx * vec_batch_stride
        + n_head_idx * vec_n_head_stride
        + vec_cols_stride * vec_arange[None, :]
    )
    vec_mask = vec_arange[None, :] < vec_cols
    vec = tl.load(pointer=vec_ptr, mask=vec_mask, other=0.0).to(tl.float32)
    if SCALER != 1:
        vec = vec * SCALER

    if VEC_SOFTMAX:
        vec_max = tl.max(vec, axis=1)
        vec = vec - vec_max[None, :]
        vec = tl.exp(vec)
        vec = vec / tl.sum(vec, axis=1)[None, :]

    matrix_ptr = matrix_tensor + (
        batch_idx * matrix_batch_stride
        + n_head_idx * matrix_n_head_stride
        + vec_arange[None, :] * matrix_rows_stride  # cols
        + (n_block_idx * SIZE_N + size_n_arange)[:, None] * matrix_cols_stride  # rows
    )

    matrix_mask = (vec_arange[None, :] < mat_rows) & ((n_block_idx * SIZE_N + size_n_arange)[:, None] < mat_cols)

    mat = tl.load(pointer=matrix_ptr, mask=matrix_mask, other=0.0).to(tl.float32)

    result = vec * mat
    result = tl.sum(input=result, axis=1)

    out_ptr = out_tensor + (
        batch_idx * out_batch_stride
        + n_head_idx * out_n_head_stride
        + (n_block_idx * SIZE_N + size_n_arange) * out_cols_stride
    )
    out_mask = (n_block_idx * SIZE_N + size_n_arange) < out_cols

    tl.store(pointer=out_ptr, value=result, eviction_policy="evict_first", mask=out_mask)


def triton_wrapper(
    vec: torch.Tensor, matrix: torch.Tensor, output: torch.Tensor, scaler: float, softmax_vec: bool, transpose_mat: bool
) -> torch.Tensor:
    matrix_stride = list(matrix.stride())
    vec_cols = vec.shape[3]
    out_cols = output.shape[3]
    if transpose_mat:
        _, _, mat_cols, mat_rows = matrix.shape
        # to transpose matrix we just need to swap strides!
        matrix_stride[-1], matrix_stride[-2] = matrix_stride[-2], matrix_stride[-1]
    else:
        _, _, mat_rows, mat_cols = matrix.shape
        if matrix_stride[-1] == 1:  # is row major
            # change layout to col major
            matrix.set_(source=matrix.permute(0, 1, 3, 2).contiguous().permute(0, 1, 3, 2))

    assert vec.shape[2] == output.shape[2] == 1
    assert mat_cols == out_cols
    assert vec_cols == mat_rows

    def grid(args) -> Tuple[int, int, int]:
        return (triton.cdiv(mat_cols, args["SIZE_N"]), batch_size, n_head)

    vec_cols_pow_2 = triton.next_power_of_2(vec_cols)

    mul_vec_mat[grid](
        vec_cols,
        mat_rows,
        mat_cols,
        out_cols,
        vec,
        matrix,
        output,
        *vec.stride(),
        *matrix_stride,
        *output.stride(),
        scaler,
        vec_cols_pow_2,
        softmax_vec,
    )
    return output


def reference_pytorch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    output_qkt: torch.Tensor,
    output_qktv: torch.Tensor,
    scaler: float,
) -> torch.Tensor:
    output_qkt = torch.matmul(input=q, other=k.transpose(-1, -2), out=output_qkt)
    p = torch.nn.functional.softmax(output_qkt * scaler, dim=-1, dtype=torch.float32).half()
    output_qktv = torch.matmul(input=p, other=v, out=output_qktv)
    return output_qktv


batch_size = 5
n_head = 20
seq_len_k = 1500
d_head = 64

cache = torch.empty(int(256e6), dtype=torch.int8, device="cuda")
q = torch.randn((batch_size, n_head, 1, d_head), dtype=torch.float16, device="cuda")
k = torch.randn((batch_size, n_head, seq_len_k, d_head), dtype=torch.float16, device="cuda")
out_qkt = torch.zeros((batch_size, n_head, 1, seq_len_k), dtype=torch.float16, device="cuda")
expected_qkt = torch.zeros_like(out_qkt)

qkt = torch.randn((batch_size, n_head, 1, seq_len_k), dtype=torch.float16, device="cuda")
v = torch.randn((batch_size, n_head, seq_len_k, d_head), dtype=torch.float16, device="cuda")
v_triton = torch.clone(v)
out_qktv = torch.zeros((batch_size, n_head, 1, d_head), dtype=torch.float16, device="cuda")
expected_qktv = torch.zeros_like(out_qktv)

print("CUDA times")
n_repeat = 1000
start_event = [torch.cuda.Event(enable_timing=True) for _ in range(n_repeat)]
end_event = [torch.cuda.Event(enable_timing=True) for _ in range(n_repeat)]


def triton_fn():
    triton_wrapper(vec=q, matrix=k, output=out_qkt, softmax_vec=False, transpose_mat=True, scaler=1.0)
    triton_wrapper(vec=out_qkt, matrix=v_triton, output=out_qktv, softmax_vec=True, transpose_mat=False, scaler=0.3)


def ref_fn():
    reference_pytorch(q=q, k=k, v=v, output_qkt=expected_qkt, output_qktv=expected_qktv, scaler=0.3)


triton_cg = cuda_graphs_wrapper(triton_fn, [])
ref_cg = cuda_graphs_wrapper(ref_fn, [])
nothing_cg = cuda_graphs_wrapper(lambda: None, [])


# warmup
for _ in range(n_repeat):
    triton_cg()
    ref_cg()
    nothing_cg()

torch.cuda.synchronize()

# run PyTorch QK^t
for i in range(n_repeat):
    cache.zero_()
    torch.cuda.synchronize()
    start_event[i].record()
    ref_cg()
    end_event[i].record()

times_pytorch_qkt = torch.median(torch.tensor([s.elapsed_time(e) for s, e in zip(start_event, end_event)]))

# run Triton QK^t
for i in range(n_repeat):
    cache.zero_()
    torch.cuda.synchronize()
    start_event[i].record()
    triton_cg()
    end_event[i].record()

times_triton_t_qkt = torch.median(torch.tensor([s.elapsed_time(e) for s, e in zip(start_event, end_event)]))

# run PyTorch QK^tV
for i in range(n_repeat):
    cache.zero_()
    torch.cuda.synchronize()
    start_event[i].record()
    nothing_cg()
    end_event[i].record()

times_overhead = torch.median(torch.tensor([s.elapsed_time(e) for s, e in zip(start_event, end_event)]))


assert torch.sum(out_qkt) != 0
assert torch.sum(out_qktv) != 0
assert torch.allclose(out_qkt, expected_qkt, atol=1e-1), f"{out_qkt}\n{expected_qkt}"
assert torch.allclose(out_qktv, expected_qktv, atol=1e-1), f"{out_qktv}\n{expected_qktv}"
print(f"{'Triton':<25}{times_triton_t_qkt.item() :.3f}")
print(f"{'PyTorch':<25}{times_pytorch_qkt.item() :.3f}")
print(f"{'overhead CUDA graphs':<25}{times_overhead.item() :.3f}")
print(f"ratio {(times_pytorch_qkt - times_overhead).item() / (times_triton_t_qkt - times_overhead).item():.3f}")
