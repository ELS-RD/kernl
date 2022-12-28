from typing import Tuple

import torch
import triton
import triton.language as tl
from torch.autograd.function import FunctionCtx
from torch.cuda.amp import custom_fwd


@triton.autotune(
    configs=[
        triton.Config({"SIZE_N": 64}, num_warps=1, num_stages=1),
        triton.Config({"SIZE_N": 32}, num_warps=1, num_stages=1),
        triton.Config({"SIZE_N": 16}, num_warps=1, num_stages=1),
        triton.Config({"SIZE_N": 8}, num_warps=1, num_stages=1),
        triton.Config({"SIZE_N": 4}, num_warps=1, num_stages=1),
        triton.Config({"SIZE_N": 2}, num_warps=1, num_stages=1),
        triton.Config({"SIZE_N": 1}, num_warps=1, num_stages=1),
    ],
    key=["size_vec_cols", "size_mat_rows", "size_mat_cols", "size_out_cols"],
)
@triton.jit
def vec_mat(
    size_vec_cols: tl.constexpr,
    size_mat_rows: tl.constexpr,
    size_mat_cols: tl.constexpr,
    size_out_cols: tl.constexpr,
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
    SHOULD_VEC_SOFTMAX: tl.constexpr,
    VEC_COLS: tl.constexpr,
    SIZE_N: tl.constexpr,
):
    n_block_idx = tl.program_id(0)
    n_head_idx = tl.program_id(1)
    batch_idx = tl.program_id(2)

    range_offs_size_n = tl.arange(0, SIZE_N)
    range_offs_vec = tl.arange(0, VEC_COLS)

    ptrs_vec = vec_tensor + (
        batch_idx * vec_batch_stride + n_head_idx * vec_n_head_stride + vec_cols_stride * range_offs_vec[:, None]
    )
    vec_mask = range_offs_vec[:, None] < size_vec_cols
    vec = tl.load(pointer=ptrs_vec, mask=vec_mask, other=0.0).to(tl.float32)

    if SCALER != 1.0:
        vec = vec * SCALER

    if SHOULD_VEC_SOFTMAX:
        vec_max = tl.max(vec, axis=0)
        vec = vec - vec_max[:, None]
        vec = tl.exp(vec)
        vec = vec / tl.sum(vec, axis=0)[:, None]

    ptrs_matrix = matrix_tensor + (
        batch_idx * matrix_batch_stride
        + n_head_idx * matrix_n_head_stride
        + range_offs_vec[:, None] * matrix_rows_stride  # cols
        + (n_block_idx * SIZE_N + range_offs_size_n)[None, :] * matrix_cols_stride  # rows
    )
    matrix_mask = (range_offs_vec[:, None] < size_mat_rows) & (
        (n_block_idx * SIZE_N + range_offs_size_n)[None, :] < size_mat_cols
    )
    mat = tl.load(pointer=ptrs_matrix, mask=matrix_mask, other=0.0).to(tl.float32)
    result = vec * mat
    result = tl.sum(input=result, axis=0)

    ptrs_out = out_tensor + (
        batch_idx * out_batch_stride
        + n_head_idx * out_n_head_stride
        + (n_block_idx * SIZE_N + range_offs_size_n) * out_cols_stride
    )
    out_mask = (n_block_idx * SIZE_N + range_offs_size_n) < size_out_cols
    tl.store(pointer=ptrs_out, value=result, mask=out_mask, eviction_policy="evict_first")


def vec_mat_wrapper(
    vec: torch.Tensor,
    matrix: torch.Tensor,
    output: torch.Tensor,
    scaler: float,
    softmax_vec: bool,
    transpose_mat: bool,
) -> torch.Tensor:
    """
    Matrix multiplication of a vector and a matrix:
    Oi = sum_j Vj * Mij
    If transpose_mat is True, the matrix is transposed:
    Oi = sum_j Vj * Mji
    If softmax_vec is True, the vector is softmaxed:
    Oi = sum_j exp(Vj - max(V)) / sum_k exp(Vk - max(V)) * Mij

    @param vec: vector to multiply with the matrix
    @param matrix: matrix to multiply with the vector
    @param output: output tensor
    @param scaler: scaler to multiply the vector with before multiplication
    @param softmax_vec: whether to softmax the vector before multiplication
    @param transpose_mat: whether to transpose the matrix before multiplication
    @return: output tensor
    """
    vec_cols = vec.shape[-1]
    out_cols = output.shape[-1]

    batch, heads, mat_rows, mat_cols = matrix.shape
    matrix_stride = list(matrix.stride())
    if transpose_mat:
        matrix_stride[-1], matrix_stride[-2] = matrix_stride[-2], matrix_stride[-1]
        mat_rows, mat_cols = mat_cols, mat_rows

    assert vec.shape[-2] == output.shape[-2] == 1
    assert mat_cols == out_cols
    assert vec_cols == mat_rows

    def grid(args) -> Tuple[int, int, int]:
        return triton.cdiv(mat_cols, args["SIZE_N"]), heads, batch

    vec_cols_pow_2 = triton.next_power_of_2(vec_cols)

    vec_mat[grid](
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
        softmax_vec,
        vec_cols_pow_2,
    )
    return output


class AttentionVecMat(torch.autograd.Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float16)
    def forward(
        ctx: FunctionCtx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        output: torch.Tensor,
        sm_scale: float,
    ) -> torch.Tensor:
        assert q.shape[2] == 1, f"q must be 1d on dim 2 but is {q.shape[2]}"
        assert v.shape == k.shape
        batch_size, n_head, seq_len_k, d_head = k.shape
        out_qkt = torch.empty((batch_size, n_head, 1, seq_len_k), dtype=torch.float32, device="cuda")
        vec_mat_wrapper(vec=q, matrix=k, output=out_qkt, softmax_vec=False, transpose_mat=True, scaler=1.0)
        vec_mat_wrapper(vec=out_qkt, matrix=v, output=output, softmax_vec=True, transpose_mat=False, scaler=sm_scale)
        return output


def attention_vec_mat_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    output: torch.Tensor,
    sm_scale: float,
    is_causal: bool = False,
    attention_mask: None = None,
):
    assert is_causal is False, "causal mask is not supported"
    assert attention_mask is None, "attention_mask is not supported"
    assert output.shape == q.shape[:3] + k.shape[-1:], f"{output.shape} != {q.shape[:3] + k.shape[-1:]}"
    assert v.stride(2) == 1, f"v must have stride == 1 on dim 2 but is {v.stride(2)}"
    return AttentionVecMat.apply(q, k, v, output, sm_scale)
