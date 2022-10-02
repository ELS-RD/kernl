import torch
import triton
import triton.language as tl


# CREDITS: Initially inspired by the Triton tutorial


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_stages=2, num_warps=8
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_stages=2, num_warps=8
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_stages=2, num_warps=4
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_stages=2, num_warps=4
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_stages=2, num_warps=4
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_stages=3, num_warps=4
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_stages=3, num_warps=4
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_stages=3, num_warps=4
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_stages=5, num_warps=2
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_stages=5, num_warps=2
        ),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def matmul_kernel(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    c_ptr,
    # Matrix dimensions
    M,
    N,
    K,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. stride_am is how much to increase a_ptr
    # by to get the element one row down (A has M rows)
    batch_stride_a,
    stride_am,
    stride_ak,
    batch_stride_b,
    stride_bk,
    stride_bn,
    batch_stride_c,
    stride_cm,
    stride_cn,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse
    # See above `L2 Cache Optimizations` section for details

    # Supergrouping of blocks
    # To see later
    batch_id = tl.program_id(axis=1)
    # program ID
    pid = tl.program_id(axis=0)
    # number of program ids along the M axis
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    # number of programs ids along the N axis
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    # number of programs in group
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    # id of the group this program is in
    group_id = pid // num_pid_in_group
    # row-id of the first program in the group
    first_pid_m = group_id * GROUP_SIZE_M
    # if `num_pid_m` isn't divisible by `GROUP_SIZE_M`, the last group is smaller
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    # *within groups*, programs are ordered in a column-major order
    # row-id of the program in the *launch grid*
    pid_m = first_pid_m + (pid % group_size_m)
    # col-id of the program in the *launch grid*
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # a_ptrs is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # b_ptrs is a block of [BLOCK_SIZE_K, BLOCK_SIZE_n] pointers
    # see above `Pointer Arithmetics` section for details

    # pid_m * BLOCK_SIZE_M is the row index of the first element of the block of size BLOCK_SIZE_M
    # We add tl.arange(0, BLOCK_SIZE_M) to get a vector of row indexes
    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # offs_am[:, None] is a column vector of BLOCK_SIZE_M rows indexes
    # We multiply by stride_am, to we get a column vector of memory offsets to each start of a row
    # offs_k[None, :] is a row vector of size BLOCK_SIZE_K columns indexes
    # We multiply stride_ak to get a row vector of memory offsets to each start of a column
    # When we add both. We get a matrix of memory offsets.
    # For A in RowMajor stride_ak will be 1, so offs_k[None, :] * stride_ak will be just 0,1,2,3,4,5....BLOCK_SIZE_K
    a_ptrs = a_ptr + batch_stride_a * batch_id + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + batch_stride_b * batch_id + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_SIZE_K):
        # Note that for simplicity, we don't apply a mask here.
        # This means that if K is not a multiple of BLOCK_SIZE_K,
        # this will access out-of-bounds memory and produce an
        # error or (worse!) incorrect results.
        a_mask = (offs_am[:, None] < M) & (offs_k[None, :] < K)
        b_mask = (offs_k[:, None] < K) & (offs_bn[None, :] < N)
        a = tl.load(a_ptrs, mask=a_mask, other=0)
        b = tl.load(b_ptrs, mask=b_mask, other=0)
        # We accumulate along the K dimension
        accumulator += tl.dot(a, b)
        # Advance the ptrs to the next K block
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    c = accumulator.to(tl.float16)

    # -----------------------------------------------------------
    # Write back the block of the output matrix C
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + batch_stride_c * batch_id + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def batched_matmul(a, b):
    # checks constraints
    assert a.shape[2] == b.shape[1], "incompatible dimensions"
    assert a.is_contiguous(), "matrix A must be contiguous"
    assert b.is_contiguous(), "matrix B must be contiguous"
    batch_size, M, K = a.shape
    _, K, N = b.shape
    # assert (
    #         K % 32 == 0
    # ), "We don't check memory-out-of-bounds with K so K must be divisible by BLOCK_SIZE_K"
    # allocates output
    c = torch.empty((batch_size, M, N), device=a.device, dtype=a.dtype)
    # 1D launch kernel where each block gets its own program.

    grid = lambda META: (  # noqa: E731
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
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
