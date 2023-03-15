from typing import Tuple

import torch
import triton
import triton.language as tl
from torch.autograd.function import FunctionCtx
from torch.cuda.amp import custom_fwd


@triton.autotune(
    configs=[
        triton.Config({"N_SIZE": 64}, num_warps=1, num_stages=1),
        triton.Config({"N_SIZE": 32}, num_warps=1, num_stages=1),
        triton.Config({"N_SIZE": 16}, num_warps=1, num_stages=1),
        triton.Config({"N_SIZE": 8}, num_warps=1, num_stages=1),
        triton.Config({"N_SIZE": 4}, num_warps=1, num_stages=1),
        triton.Config({"N_SIZE": 2}, num_warps=1, num_stages=1),
        triton.Config({"N_SIZE": 1}, num_warps=1, num_stages=1),
    ],
    key=["vec_col_size", "matrix_row_size", "matrix_col_size", "output_col_size"],
)
@triton.jit
def vec_mat(
    vec_col_size: tl.constexpr,
    matrix_row_size: tl.constexpr,
    matrix_col_size: tl.constexpr,
    output_col_size: tl.constexpr,
    vec_ptr,
    vec_batch_stride,
    vec_head_stride,
    vec_row_stride,
    vec_col_stride,
    matrix_ptr,
    matrix_batch_stride,
    matrix_head_stride,
    matrix_row_stride,
    matrix_col_stride,
    output_ptr,
    output_batch_stride,
    output_head_stride,
    output_row_stride,
    output_col_stride,
    SCALER: tl.constexpr,
    SHOULD_VEC_SOFTMAX: tl.constexpr,
    VEC_COL_ROUNDED_SIZE: tl.constexpr,
    N_SIZE: tl.constexpr,
):
    block_n_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    batch_idx = tl.program_id(2)

    n_range_offs = tl.arange(0, N_SIZE)
    vec_col_rounded_range_offs = tl.arange(0, VEC_COL_ROUNDED_SIZE)

    vec_ptrs = vec_ptr + (
        batch_idx * vec_batch_stride + head_idx * vec_head_stride + vec_col_stride * vec_col_rounded_range_offs[:, None]
    )
    vec_ptr_mask = vec_col_rounded_range_offs[:, None] < vec_col_size
    vec = tl.load(pointer=vec_ptrs, mask=vec_ptr_mask, other=0.0).to(tl.float32)

    if SCALER != 1.0:
        vec = vec * SCALER

    if SHOULD_VEC_SOFTMAX:
        vec_max = tl.max(vec, axis=0)
        vec = vec - vec_max[:, None]
        vec = tl.exp(vec)
        vec = vec / tl.sum(vec, axis=0)[:, None]

    matrix_ptrs = matrix_ptr + (
        batch_idx * matrix_batch_stride
        + head_idx * matrix_head_stride
        + vec_col_rounded_range_offs[:, None] * matrix_row_stride  # cols
        + (block_n_idx * N_SIZE + n_range_offs)[None, :] * matrix_col_stride  # rows
    )
    matrix_ptr_mask = (vec_col_rounded_range_offs[:, None] < matrix_row_size) & (
        (block_n_idx * N_SIZE + n_range_offs)[None, :] < matrix_col_size
    )
    matrix = tl.load(pointer=matrix_ptrs, mask=matrix_ptr_mask, other=0.0).to(tl.float32)

    result = vec * matrix
    result = tl.sum(input=result, axis=0)

    output_ptrs = output_ptr + (
        batch_idx * output_batch_stride
        + head_idx * output_head_stride
        + (block_n_idx * N_SIZE + n_range_offs) * output_col_stride
    )
    output_ptr_mask = (block_n_idx * N_SIZE + n_range_offs) < output_col_size
    tl.store(pointer=output_ptrs, value=result, mask=output_ptr_mask)


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
        return triton.cdiv(mat_cols, args["N_SIZE"]), heads, batch

    vec_cols_pow_2 = triton.next_power_of_2(vec_cols)

    vec_mat[grid](
        vec_cols,
        mat_rows,
        mat_cols,
        out_cols,
        vec,
        *vec.stride(),
        matrix,
        *matrix_stride,
        output,
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

    return AttentionVecMat.apply(q, k, v, output, sm_scale)
