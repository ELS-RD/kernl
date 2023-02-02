import torch
import triton
import triton.language as tl

from conftest import assert_all_close
from src.kernl.debugger.debugger import triton_debug_autotune

from kernl.debugger.debugger import triton_debug
from kernl.implementations.attention import attention_reference


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


def test_attention():
    batch, heads, len_ctxt, d_head = 1, 2, 32, 64
    q = torch.empty((batch, heads, len_ctxt, d_head), device="cuda", dtype=torch.float16).normal_(mean=0, std=0.5)
    k = torch.empty_like(q).normal_(mean=0, std=0.5)
    v = torch.empty_like(q).normal_(mean=0, std=0.5)
    sm_scale = 0.3
    output = torch.zeros_like(q)
    reference_output = torch.zeros_like(output)
    tmp = torch.zeros((batch, heads, len_ctxt), device="cuda", dtype=torch.float32)

    @triton_debug_autotune(
        configs=[
            triton.Config({"BLOCK_M": 16, "BLOCK_N": 16}, num_stages=1, num_warps=4),
        ],
        key=["size_m_cache_key", "size_n_cache_key", "heads", "HAS_MASK", "IS_MATRIX_MASK", "IS_CAUSAL"],
    )
    @triton_debug
    def _fwd_kernel(
        heads,
        size_m,
        size_n,
        size_m_cache_key,
        size_n_cache_key,
        Q,
        K,
        V,
        sm_scale,
        attention_mask,
        TMP,  # NOTE: TMP is a scratchpad buffer to workaround a compiler bug
        output,
        q_batch_stride,
        q_head_stride,
        q_m_stride,
        q_k_stride,
        k_batch_stride,
        k_head_stride,
        k_n_stride,
        k_k_stride,  # axis named n,k instead of k,n because of the transpose of K matrix
        v_batch_stride,
        v_head_stride,
        v_k_stride,
        v_n_stride,
        o_batch_stride,
        o_head_stride,
        o_m_stride,
        o_n_stride,
        attention_mask_batch_stride,
        attention_mask_head_stride,
        attention_mask_m_stride,
        attention_mask_n_stride,
        min_clamp_value,
        attention_mask_batch_size,
        attention_mask_head_size,
        attention_mask_m_size,
        attention_mask_n_size,
        tmp_batch_size,
        tmp_head_size,
        tmp_m_size,
        HAS_MASK: tl.constexpr,
        IS_MATRIX_MASK: tl.constexpr,
        IS_CAUSAL: tl.constexpr,
        BLOCK_DHEAD: tl.constexpr,
        BLOCK_M: tl.constexpr,  # this parameter and below are managed by the autotune and need to be at the end
        BLOCK_N: tl.constexpr,
        NEED_LOAD_MASK_SIZE_M: tl.constexpr,
        NEED_LOAD_MASK_SIZE_N: tl.constexpr,
    ):
        # Index of the block on M axis (M axis is the rows of matrix K)
        m_block_idx = tl.program_id(0)
        # Global index of the current head
        head_idx = tl.program_id(1)

        # offsets
        range_offs_m = tl.arange(0, BLOCK_M)  # first block on M dimension
        range_offs_n = tl.arange(0, BLOCK_N)  # first block on N dimension
        range_offs_d = tl.arange(0, BLOCK_DHEAD)  # full head

        offs_m = m_block_idx * BLOCK_M + range_offs_m  # rows offsets on M axis

        current_batch_idx = head_idx // heads
        current_head_idx = head_idx % heads

        # memory offsets matrices on whole Q, K, V, output matrices
        # offsets for the current block on matrix Q
        offs_q = (
            current_batch_idx * q_batch_stride
            + current_head_idx * q_head_stride
            + (offs_m[:, None] * q_m_stride + range_offs_d[None, :] * q_k_stride)
        )

        # offsets for the first block on matrix K
        offs_k = (
            current_batch_idx * k_batch_stride
            + current_head_idx * k_head_stride
            + (range_offs_n[:, None] * k_n_stride + range_offs_d[None, :] * k_k_stride)
        )

        # offsets for the first block on matrix V
        offs_v = (
            current_batch_idx * v_batch_stride
            + current_head_idx * v_head_stride
            + (range_offs_n[:, None] * v_k_stride + range_offs_d[None, :] * v_n_stride)
        )

        # offsets for the current block on matrix Output
        offs_o = (
            current_batch_idx * o_batch_stride
            + current_head_idx * o_head_stride
            + (offs_m[:, None] * o_m_stride + range_offs_d[None, :] * o_n_stride)
        )

        # pointers to blocks in Q, K, V
        ptrs_q = Q + offs_q
        ptrs_k = K + offs_k
        ptrs_v = V + offs_v
        ptrs_o = output + offs_o

        # Temporary pointer to memory to fix bug in triton compiler
        ptrs_t = TMP + tmp_m_size * head_idx + offs_m

        # initialize pointer to m and d used to compute normalizer for softmax
        l_i = tl.zeros((BLOCK_M,), dtype=tl.float32) - float("inf")
        d_i = tl.zeros((BLOCK_M,), dtype=tl.float32)

        # initialize the main loop accumulator, it is the size of a block of full rows, written to the output for the
        # current head
        acc = tl.zeros((BLOCK_M, BLOCK_DHEAD), dtype=tl.float32)

        # load q, a block of full rows of matrix q
        # there is a bug on size_n, it is not related to Q tensor but if a load mask is needed and BLOCK_N > size_n,
        # output is wrong.
        if NEED_LOAD_MASK_SIZE_M | NEED_LOAD_MASK_SIZE_N:
            q = tl.load(ptrs_q, mask=offs_m[:, None] < size_m, other=0.0)
        else:
            q = tl.load(ptrs_q)

        n_end = size_n
        if IS_CAUSAL:
            n_end = (m_block_idx + 1) * BLOCK_M

        if HAS_MASK:
            mask_batch_idx = (current_batch_idx,)
            if attention_mask_batch_size == 1:
                mask_batch_idx = 0

            mask_head_idx = current_head_idx
            if attention_mask_head_size == 1:
                mask_head_idx = 0

            offs_base_mask = mask_batch_idx * attention_mask_batch_stride + mask_head_idx * attention_mask_head_stride

        for block_start_index_n in range(0, n_end, BLOCK_N):
            block_start_index_n = tl.multiple_of(block_start_index_n, BLOCK_N)
            offs_n = block_start_index_n + range_offs_n
            # We load the current block in K in SRAM
            # We do the first multiplication between the block in Q and the current block in K
            # We finish with the scaling (sqrt(BLOCK_DHEAD) in Vaswani et al. but sm_scale here)
            if NEED_LOAD_MASK_SIZE_N:
                load_mask = offs_n[:, None] < size_n
                k = tl.load(ptrs_k + block_start_index_n * k_n_stride, mask=load_mask, other=0.0)
            else:
                k = tl.load(ptrs_k + block_start_index_n * k_n_stride)
            qk = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

            # required to fix a Triton compiler bug, if not done, there is a precision issue
            if NEED_LOAD_MASK_SIZE_N:
                qk = tl.where(range_offs_n[None, :] < size_n, qk, float("-inf"))

            qk += tl.dot(q, k, trans_b=True)
            qk *= sm_scale
            if IS_CAUSAL:
                qk += tl.where(offs_m[:, None] >= offs_n[None, :], 0, float("-inf"))

            if HAS_MASK:
                # we assume mask has a vector shape
                offs_mask = offs_base_mask + offs_n[None, :] * attention_mask_n_stride
                if IS_MATRIX_MASK:  # mask has a matrix shape, we load (BLOCK_M, BLOCK_N) elements
                    offs_mask += offs_m[:, None] * attention_mask_m_stride

                if NEED_LOAD_MASK_SIZE_N & (not IS_MATRIX_MASK):  # mask has a vector shape + need a load mask
                    attention_load_mask = offs_n[None, :] < attention_mask_n_size
                if IS_MATRIX_MASK:  # mask has a matrix shape
                    if NEED_LOAD_MASK_SIZE_M & (not NEED_LOAD_MASK_SIZE_N):  # load mask on M axis
                        attention_load_mask = offs_m[:, None] < attention_mask_m_size
                    elif (not NEED_LOAD_MASK_SIZE_M) & NEED_LOAD_MASK_SIZE_N:  # load mask on N axis
                        attention_load_mask = offs_n[None, :] < attention_mask_n_size
                    elif NEED_LOAD_MASK_SIZE_M & NEED_LOAD_MASK_SIZE_N:  # load mask on both axis
                        attention_load_mask = (offs_n[None, :] < attention_mask_n_size) & (
                            offs_m[:, None] < attention_mask_m_size
                        )

                if (NEED_LOAD_MASK_SIZE_M & IS_MATRIX_MASK) | NEED_LOAD_MASK_SIZE_N:
                    m = tl.load(
                        attention_mask + offs_mask,
                        eviction_policy="evict_first",
                        mask=attention_load_mask,
                        other=float("-inf"),
                    )
                else:
                    m = tl.load(
                        attention_mask + offs_mask,
                        eviction_policy="evict_first",
                    )
                # Avoids NaN
                m = tl.where(m == float("-inf"), min_clamp_value, m)
                qk += m

            # We compute softmax normalization like in Milakov et al.
            # We renamed m (in the original article) to l to avoid confusions
            # We start with the current block qk
            l_j = tl.max(qk, 1)

            numerators = tl.exp(qk - l_j[:, None])
            d_j = tl.sum(numerators, 1)

            l_new = tl.maximum(l_i, l_j)
            alpha = tl.exp(l_i - l_new)
            beta = tl.exp(l_j - l_new)
            d_new = alpha * d_i + beta * d_j

            p_scale = beta / d_new

            qk_softmax = numerators * p_scale[:, None]

            # From here, qk_softmax is correct related to all over previously done block
            # However it is wrong related to the full output row if the qk isn't the last block
            # And at this stage all the previously done qk_softmax blocks are also wrong and needs to be corrected
            # To correct previous blocks we will scale the accumulator
            # d_i / d_new is for correcting denominator
            # alpha is for correcting numerator
            acc_scale = d_i / d_new * alpha

            # This isn't useful in the algorithm, simply to fix a compiler bug
            # BUG: have to store and immediately load
            tl.store(ptrs_t, acc_scale)
            acc_scale = tl.load(ptrs_t)

            # acc scaling
            acc = acc * acc_scale[:, None]

            # We now apply the last operation, the multiplication by a block of matrix V
            if NEED_LOAD_MASK_SIZE_N:
                load_mask = offs_n[:, None] < size_n  # repeated otherwise triton segfault
                v = tl.load(ptrs_v + block_start_index_n * v_k_stride, mask=load_mask, other=0.0)
            else:
                v = tl.load(ptrs_v + block_start_index_n * v_k_stride)
            qk_softmax = qk_softmax.to(tl.float16)
            acc += tl.dot(qk_softmax, v)

            # We update the normalizer for the next iteration
            d_i = d_new
            l_i = l_new

        if NEED_LOAD_MASK_SIZE_M:
            out_store_mask = offs_m[:, None] < size_m
            tl.store(ptrs_o, acc, mask=out_store_mask)
        else:
            tl.store(ptrs_o, acc)

    grid = lambda args: (triton.cdiv(len_ctxt, args["BLOCK_M"]), batch * heads)  # noqa: E731
    _fwd_kernel[grid](
        heads=heads,
        size_m=len_ctxt,
        size_n=len_ctxt,
        size_m_cache_key=len_ctxt,
        size_n_cache_key=len_ctxt,
        Q=q,
        K=k,
        V=v,
        sm_scale=sm_scale,
        attention_mask=None,
        TMP=tmp,  # NOTE: TMP is a scratchpad buffer to workaround a compiler bug
        output=output,
        q_batch_stride=q.stride(0),
        q_head_stride=q.stride(1),
        q_m_stride=q.stride(2),
        q_k_stride=q.stride(3),
        k_batch_stride=k.stride(0),
        k_head_stride=k.stride(1),
        k_n_stride=k.stride(2),
        k_k_stride=k.stride(3),
        v_batch_stride=v.stride(0),
        v_head_stride=v.stride(1),
        v_k_stride=v.stride(2),
        v_n_stride=v.stride(3),
        o_batch_stride=output.stride(0),
        o_head_stride=output.stride(1),
        o_m_stride=output.stride(2),
        o_n_stride=output.stride(3),
        attention_mask_batch_stride=0,
        attention_mask_head_stride=0,
        attention_mask_m_stride=0,
        attention_mask_n_stride=0,
        min_clamp_value=0,
        attention_mask_batch_size=0,
        attention_mask_head_size=0,
        attention_mask_m_size=0,
        attention_mask_n_size=0,
        tmp_batch_size=tmp.shape[0],
        tmp_head_size=tmp.shape[1],
        tmp_m_size=tmp.shape[2],
        HAS_MASK=False,
        IS_MATRIX_MASK=False,
        IS_CAUSAL=False,
        BLOCK_DHEAD=d_head,
        NEED_LOAD_MASK_SIZE_M=False,
        NEED_LOAD_MASK_SIZE_N=False,
    )

    attention_reference(q=q, k=k, v=v, output=reference_output, sm_scale=sm_scale, is_causal=False, attention_mask=None)
    assert torch.allclose(output, reference_output, atol=1e-1)
