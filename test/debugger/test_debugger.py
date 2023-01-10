import torch
import triton
import triton.language as tl

from conftest import assert_all_close
from src.kernl.debugger.debugger import triton_debug_autotune

from kernl.debugger.debugger import triton_debug


def test_addition():
    @triton_debug
    def add_kernel(
        x_ptr,  # *Pointer* to first input vector
        y_ptr,  # *Pointer* to second input vector
        output_ptr,  # *Pointer* to output vector
        n_elements,  # Size of the vector
        BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process
        # NOTE: `constexpr` so it can be used as a shape value
    ):
        # There are multiple 'program's processing different data. We identify which program
        # we are here
        pid = tl.program_id(axis=0)  # We use a 1D launch grid so axis is 0
        # This program will process inputs that are offset from the initial data.
        # for instance, if you had a vector of length 256 and block_size of 64, the programs
        # would each access the elements [0:64, 64:128, 128:192, 192:256].
        # Note that offsets is a list of pointers
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        # Create a mask to guard memory operations against out-of-bounds accesses
        mask = offsets < n_elements
        # Load x and y from DRAM, masking out any extra elements in case the input is not a
        # multiple of the block size
        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)
        output = x + y
        # Write x + y back to DRAM
        tl.store(output_ptr + offsets, output, mask=mask)

    a = torch.rand((128,), device="cuda")
    b = torch.rand((128,), device="cuda")
    expected = a + b

    output = torch.empty((128,), device="cuda")

    def grid(meta):
        return (triton.cdiv(128, meta["BLOCK_SIZE"]),)

    add_kernel[grid](a, b, output, 128, BLOCK_SIZE=32)

    assert_all_close(expected, output)


def test_softmax():
    @triton_debug
    def softmax_kernel(output_ptr, input_ptr, input_row_stride, output_row_stride, n_cols, BLOCK_SIZE: tl.constexpr):
        # The rows of the softmax are independent, so we parallelize across those
        row_idx = tl.program_id(0)
        # The stride represents how much we need to increase the pointer to advance 1 row
        row_start_ptr = input_ptr + row_idx * input_row_stride
        # The block size is the next power of two greater than n_cols, so we can fit each
        # row in a single block
        col_offsets = tl.arange(0, BLOCK_SIZE)
        input_ptrs = row_start_ptr + col_offsets
        # Load the row into SRAM, using a mask since BLOCK_SIZE may be > than n_cols
        row = tl.load(input_ptrs, mask=col_offsets < n_cols, other=-float("inf"))
        # Substract maximum for numerical stability
        row_minus_max = row - tl.max(row, axis=0)
        # Note that exponentials in Triton are fast but approximate (i.e., think __expf in CUDA)
        numerator = tl.exp(row_minus_max)
        denominator = tl.sum(numerator, axis=0)
        softmax_output = numerator / denominator
        # Write back output to DRAM
        output_row_start_ptr = output_ptr + row_idx * output_row_stride
        output_ptrs = output_row_start_ptr + col_offsets
        tl.store(output_ptrs, softmax_output, mask=col_offsets < n_cols)

    x = torch.rand((128, 128), device="cuda")
    expected = torch.softmax(x, axis=1)

    output = torch.empty((128, 128), device="cuda")
    BLOCK_SIZE = triton.next_power_of_2(128)
    softmax_kernel[(128,)](output, x, x.stride(0), output.stride(0), 128, num_warps=4, BLOCK_SIZE=BLOCK_SIZE)

    assert_all_close(expected, output)


def test_matmul():
    @triton_debug_autotune(
        configs=[
            triton.Config(
                {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8},
                num_stages=5,
                num_warps=2,
            ),
            triton.Config(
                {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8},
                num_stages=5,
                num_warps=2,
            ),
        ],
        key=["M", "N", "K"],
    )
    @triton_debug
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
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        stride_cm,
        stride_cn,
        # Meta-parameters
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr,
        ACTIVATION: tl.constexpr,
    ):
        """Kernel for computing the matmul C = A x B.
        A has shape (M, K), B has shape (K, N) and C has shape (M, N)
        """
        # -----------------------------------------------------------
        # Map program ids `pid` to the block of C it should compute.
        # This is done in a grouped ordering to promote L2 data reuse
        # See above `L2 Cache Optimizations` section for details
        pid = tl.program_id(axis=0)
        num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
        num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + (pid % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

        # ----------------------------------------------------------
        # Create pointers for the first blocks of A and B.
        # We will advance this pointer as we move in the K direction
        # and accumulate
        # a_ptrs is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
        # b_ptrs is a block of [BLOCK_SIZE_K, BLOCK_SIZE_n] pointers
        # see above `Pointer Arithmetics` section for details
        offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        offs_k = tl.arange(0, BLOCK_SIZE_K)
        a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

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
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
            # We accumulate along the K dimension
            accumulator += tl.dot(a, b)
            # Advance the ptrs to the next K block
            a_ptrs += BLOCK_SIZE_K * stride_ak
            b_ptrs += BLOCK_SIZE_K * stride_bk
        # you can fuse arbitrary activation functions here
        # while the accumulator is still in FP32!
        if ACTIVATION == "leaky_relu":
            accumulator = leaky_relu(accumulator)
        c = accumulator.to(tl.float16)

        # -----------------------------------------------------------
        # Write back the block of the output matrix C
        offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        tl.store(c_ptrs, c, mask=c_mask)

    def leaky_relu(x):
        x = x + 1
        return tl.where(x >= 0, x, 0.01 * x)

    def matmul(a, b, activation=""):
        # checks constraints
        assert a.shape[1] == b.shape[0], "incompatible dimensions"
        assert a.is_contiguous(), "matrix A must be contiguous"
        assert b.is_contiguous(), "matrix B must be contiguous"
        M, K = a.shape
        K, N = b.shape
        assert K % 32 == 0, "We don't check memory-out-of-bounds with K so K must be divisible by BLOCK_SIZE_K"
        # allocates output
        c = torch.empty((M, N), device=a.device, dtype=a.dtype)

        # 1D launch kernel where each block gets its own program.
        def grid(META):
            return (triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),)

        matmul_kernel[grid](
            a,
            b,
            c,
            M,
            N,
            K,
            a.stride(0),
            a.stride(1),
            b.stride(0),
            b.stride(1),
            c.stride(0),
            c.stride(1),
            ACTIVATION=activation,
        )
        return c

    a = torch.rand((512, 512), device="cuda")
    b = torch.rand((512, 512), device="cuda")
    expected = torch.matmul(a, b)

    output = matmul(a, b)

    assert_all_close(expected, output)
