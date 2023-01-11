#  Copyright 2022 Lefebvre Sarrut
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

import torch
import triton
import triton.language as tl


# CREDITS: Initially inspired by the Triton tutorial


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_M_SIZE": 128, "BLOCK_N_SIZE": 256, "BLOCK_K_SIZE": 32, "GROUP_M_SIZE": 8}, num_stages=2, num_warps=8
        ),
        triton.Config(
            {"BLOCK_M_SIZE": 256, "BLOCK_N_SIZE": 128, "BLOCK_K_SIZE": 32, "GROUP_M_SIZE": 8}, num_stages=2, num_warps=8
        ),
        triton.Config(
            {"BLOCK_M_SIZE": 256, "BLOCK_N_SIZE": 64, "BLOCK_K_SIZE": 32, "GROUP_M_SIZE": 8}, num_stages=2, num_warps=4
        ),
        triton.Config(
            {"BLOCK_M_SIZE": 64, "BLOCK_N_SIZE": 256, "BLOCK_K_SIZE": 32, "GROUP_M_SIZE": 8}, num_stages=2, num_warps=4
        ),
        triton.Config(
            {"BLOCK_M_SIZE": 128, "BLOCK_N_SIZE": 128, "BLOCK_K_SIZE": 32, "GROUP_M_SIZE": 8}, num_stages=2, num_warps=4
        ),
        triton.Config(
            {"BLOCK_M_SIZE": 128, "BLOCK_N_SIZE": 64, "BLOCK_K_SIZE": 32, "GROUP_M_SIZE": 8}, num_stages=3, num_warps=4
        ),
        triton.Config(
            {"BLOCK_M_SIZE": 64, "BLOCK_N_SIZE": 128, "BLOCK_K_SIZE": 32, "GROUP_M_SIZE": 8}, num_stages=3, num_warps=4
        ),
        triton.Config(
            {"BLOCK_M_SIZE": 128, "BLOCK_N_SIZE": 32, "BLOCK_K_SIZE": 32, "GROUP_M_SIZE": 8}, num_stages=3, num_warps=4
        ),
        triton.Config(
            {"BLOCK_M_SIZE": 64, "BLOCK_N_SIZE": 32, "BLOCK_K_SIZE": 32, "GROUP_M_SIZE": 8}, num_stages=5, num_warps=2
        ),
        triton.Config(
            {"BLOCK_M_SIZE": 32, "BLOCK_N_SIZE": 64, "BLOCK_K_SIZE": 32, "GROUP_M_SIZE": 8}, num_stages=5, num_warps=2
        ),
    ],
    key=["m_size", "n_size", "k_size"],
)
@triton.jit
def matmul_kernel(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    c_ptr,
    # Matrix dimensions
    m_size,
    n_size,
    k_size,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. stride_am is how much to increase a_ptr
    # by to get the element one row down (A has M rows)
    a_batch_stride,
    a_m_stride,
    a_k_stride,
    b_batch_stride,
    b_k_stride,
    b_n_stride,
    c_batch_stride,
    c_m_stride,
    c_n_stride,
    # Meta-parameters
    BLOCK_M_SIZE: tl.constexpr,
    BLOCK_N_SIZE: tl.constexpr,
    BLOCK_K_SIZE: tl.constexpr,
    GROUP_M_SIZE: tl.constexpr,
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    # -----------------------------------------------------------
    # Map program ids `program_idx` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse
    # See above `L2 Cache Optimizations` section for details

    # Supergrouping of blocks
    # To see later
    batch_idx = tl.program_id(axis=1)
    # program ID
    program_idx = tl.program_id(axis=0)

    # number of program ids along the M axis
    program_m_count = tl.cdiv(m_size, BLOCK_M_SIZE)
    # number of programs ids along the N axis
    program_n_count = tl.cdiv(n_size, BLOCK_N_SIZE)

    # number of programs in group
    program_in_group_count = GROUP_M_SIZE * program_n_count
    # id of the group this program is in
    group_idx = program_idx // program_in_group_count
    # row-id of the first program in the group
    first_program_m_idx = group_idx * GROUP_M_SIZE
    # if `program_m_count` isn't divisible by `GROUP_M_SIZE`, the last group is smaller
    GROUP_M_SIZE = min(program_m_count - first_program_m_idx, GROUP_M_SIZE)
    # *within groups*, programs are ordered in a column-major order
    # row-id of the program in the *launch grid*
    program_m_idx = first_program_m_idx + (program_idx % GROUP_M_SIZE)
    # col-id of the program in the *launch grid*
    program_n_idx = (program_idx % program_in_group_count) // GROUP_M_SIZE

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # a_ptrs is a block of [BLOCK_M_SIZE, BLOCK_K_SIZE] pointers
    # b_ptrs is a block of [BLOCK_K_SIZE, BLOCK_N_SIZE] pointers
    # see above `Pointer Arithmetics` section for details

    # program_m_idx * BLOCK_M_SIZE is the row index of the first element of the block of size BLOCK_M_SIZE
    # We add tl.arange(0, BLOCK_M_SIZE) to get a vector of row indexes
    a_offs = program_m_idx * BLOCK_M_SIZE + tl.arange(0, BLOCK_M_SIZE)
    b_offs = program_n_idx * BLOCK_N_SIZE + tl.arange(0, BLOCK_N_SIZE)

    k_range_offs = tl.arange(0, BLOCK_K_SIZE)

    # a_offs[:, None] is a column vector of BLOCK_M_SIZE rows indexes
    # We multiply by stride_am, to we get a column vector of memory offsets to each start of a row
    # k_range_offs[None, :] is a row vector of size BLOCK_K_SIZE columns indexes
    # We multiply stride_ak to get a row vector of memory offsets to each start of a column
    # When we add both. We get a matrix of memory offsets.
    # For A in RowMajor stride_ak will be 1, so k_range_offs[None, :] * stride_ak will be
    # just 0,1,2,3,4,5....BLOCK_K_SIZE
    a_ptrs = a_ptr + a_batch_stride * batch_idx + (a_offs[:, None] * a_m_stride + k_range_offs[None, :] * a_k_stride)
    b_ptrs = b_ptr + b_batch_stride * batch_idx + (k_range_offs[:, None] * b_k_stride + b_offs[None, :] * b_n_stride)

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix
    # We accumulate into a `[BLOCK_M_SIZE, BLOCK_N_SIZE]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop
    accumulator = tl.zeros((BLOCK_M_SIZE, BLOCK_N_SIZE), dtype=tl.float32)
    for k in range(0, k_size, BLOCK_K_SIZE):
        # Note that for simplicity, we don't apply a mask here.
        # This means that if K is not a multiple of BLOCK_K_SIZE,
        # this will access out-of-bounds memory and produce an
        # error or (worse!) incorrect results.

        a_ptr_mask = (a_offs[:, None] < m_size) & (k_range_offs[None, :] < k_size)
        a = tl.load(a_ptrs, mask=a_ptr_mask, other=0)

        b_ptr_mask = (k_range_offs[:, None] < k_size) & (b_offs[None, :] < n_size)
        b = tl.load(b_ptrs, mask=b_ptr_mask, other=0)

        # We accumulate along the K dimension
        accumulator += tl.dot(a, b)
        # Advance the ptrs to the next K block
        a_ptrs += BLOCK_K_SIZE * a_k_stride
        b_ptrs += BLOCK_K_SIZE * b_k_stride

    c = accumulator.to(tl.float16)

    # -----------------------------------------------------------
    # Write back the block of the output matrix C
    c_m_offs = program_m_idx * BLOCK_M_SIZE + tl.arange(0, BLOCK_M_SIZE)
    c_n_offs = program_n_idx * BLOCK_N_SIZE + tl.arange(0, BLOCK_N_SIZE)
    c_ptrs = c_ptr + c_batch_stride * batch_idx + c_m_stride * c_m_offs[:, None] + c_n_stride * c_n_offs[None, :]
    c_ptr_mask = (c_m_offs[:, None] < m_size) & (c_n_offs[None, :] < n_size)
    tl.store(c_ptrs, c, mask=c_ptr_mask)


def batched_matmul(a, b):
    # checks constraints
    assert a.shape[2] == b.shape[1], "incompatible dimensions"
    assert a.is_contiguous(), "matrix A must be contiguous"
    assert b.is_contiguous(), "matrix B must be contiguous"
    batch_size, M, K = a.shape
    _, K, N = b.shape
    # assert (
    #         K % 32 == 0
    # ), "We don't check memory-out-of-bounds with K so K must be divisible by BLOCK_K_SIZE"
    # allocates output
    c = torch.empty((batch_size, M, N), device=a.device, dtype=a.dtype)
    # 1D launch kernel where each block gets its own program.

    grid = lambda META: (  # noqa: E731
        triton.cdiv(M, META["BLOCK_M_SIZE"]) * triton.cdiv(N, META["BLOCK_N_SIZE"]),
        batch_size,
    )
    matmul_kernel[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        a.stride(2),
        b.stride(0),
        b.stride(1),
        b.stride(2),
        c.stride(0),
        c.stride(1),
        c.stride(2),
    )
    return c
