import torch
import triton
import triton.language as tl


@triton.jit
def _fwd_kernel(
        batch,
        heads,
        seq_length,
        Q,
        K,
        V,
        sm_scale,
        TMP,  # NOTE: TMP is a scratchpad buffer to workaround a compiler bug
        output,
        stride_qz, stride_qh, stride_qm, stride_qk,
        stride_kz, stride_kh, stride_kn, stride_kk,
        stride_vz, stride_vh, stride_vk, stride_vn,
        stride_oz, stride_oh, stride_om, stride_on,

        BLOCK_M: tl.constexpr,
        BLOCK_DHEAD: tl.constexpr,
        BLOCK_N: tl.constexpr,
):
    """
    Computes attention
    Each head is size (seq_length, BLOCK_DHEAD)

    @param batch: Batch size
    @param heads: Number of heads per batch
    @param seq_length: Sequence length
    @param Q: Query matrix size (batch, heads, seq_length, BLOCK_DHEAD)
    @param K: Key matrix size (batch, heads, seq_length, BLOCK_DHEAD)
    @param V: Value matrix size (batch, heads, seq_length, BLOCK_DHEAD)
    @param sm_scale: Scaling factor applied after operation QxK
    @param TMP: Temporary variable to fix a compiler bug
    @param output: Output matrix size (batch, heads, seq_length, BLOCK_DHEAD)
    @param stride_qz: matrix q stride for batch dimension
    @param stride_qh: matrix q stride for head dimension
    @param stride_qm: matrix q stride for rows, called "dimension m"
    @param stride_qk: matrix q stride for columns
    @param stride_kz: matrix k stride for batch dimension
    @param stride_kh: matrix k stride for head dimension
    @param stride_kn: matrix k stride for rows, called "dimension n"
    @param stride_kk: matrix k stride for columns
    @param stride_vz: matrix v stride for batch dimension
    @param stride_vh: matrix v stride for head dimension
    @param stride_vk: matrix v stride for columns
    @param stride_vn: matrix v stride for rows
    @param stride_oz: output matrix stride for batch dimension
    @param stride_oh: output matrix stride for head dimension
    @param stride_om: output matrix stride for rows
    @param stride_on: output matrix stride for columns
    @param BLOCK_M: number of rows computed in a single instance for matrix Q
    @param BLOCK_DHEAD: number of columns per head
    @param BLOCK_N:  number of rows computed at each loop in the main loop for matrix K and V
    """
    # Index of the block on M axis (M axis is the rows of matrix K)
    m_block_idx = tl.program_id(0)
    # Global index of the current head
    head_idx = tl.program_id(1)

    # offsets
    offs_m = m_block_idx * BLOCK_M + tl.arange(0, BLOCK_M)  # rows offsets on M axis
    offs_n = tl.arange(0, BLOCK_N)  # First block on N dimension
    offs_d = tl.arange(0, BLOCK_DHEAD)  # Full head

    # memory offsets matrices on whole Q, K, V matrices
    # Offsets for the current block on matrix Q
    off_q = head_idx * stride_qh + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    # Offsets for the first block on matrix K
    off_k = head_idx * stride_kh + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk
    # Offsets for the first block on matrix V
    off_v = head_idx * stride_vh + offs_n[:, None] * stride_qm + offs_d[None, :] * stride_qk

    # pointers to blocks in Q, K, V
    q_ptrs = Q + off_q
    k_ptrs = K + off_k
    v_ptrs = V + off_v

    # Temporary pointer to memory to fix bug in triton compiler
    t_ptrs = TMP + head_idx * seq_length + offs_m

    # initialize pointer to m and d used to compute normalizer for softmax
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32) - float("inf")
    d_i = tl.zeros((BLOCK_M,), dtype=tl.float32)

    # initialize the main loop accumulator, it is the size of a block of full rows, written to the output for the
    # current head
    acc = tl.zeros((BLOCK_M, BLOCK_DHEAD), dtype=tl.float32)

    # load q, a block of full rows of matrix q
    # it will stay in SRAM throughout
    q = tl.load(q_ptrs)

    # loop over k, v and update accumulator
    # n_row_offset is the row offset on dimension N of the current block
    # It's used for booth the N dimension of K and V because they are handled at the same time
    for n_row_offset in range(0, seq_length, BLOCK_N):
        n_row_offset = tl.multiple_of(n_row_offset, BLOCK_N)
        # We load the current block in K in SRAM
        # We do the first multiplication between the block in Q and the current block in K
        # We finish with the scaling (sqrt(BLOCK_DHEAD) in Vaswani et al. but sm_scale here)
        k = tl.load(k_ptrs + n_row_offset * stride_kn)
        qk = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        qk += tl.dot(q, k, trans_b=True)
        qk *= sm_scale

        # We compute softmax normalization like in Milakov et al.
        # We renamed m (in the original article) to l to avoid confusions
        # We start with the current block qk
        l_j = tl.max(qk, 1)

        p = tl.exp(qk - l_j[:, None])  # CHANGE !!! Current numerators
        d_j = tl.sum(p, 1)

        l_new = tl.maximum(l_i, l_j)
        alpha = tl.exp(l_i - l_new)
        beta = tl.exp(l_j - l_new)
        d_new = alpha * d_i + beta * d_j

        # We correct the numerator for the current softmax (*beta) -> exp(l_j - l_new) * exp(qk - mj) = exp(l_new) We
        # divide by the normalization. It's strange to do it this way instead of simply computing the softmax for qk,
        # but since all needed operations are already done for updating m and d, it seems faster
        p_scale = beta / d_new
        qk_softmax = p * p_scale[:, None]

        # From here, qk_softmax is correct related to all over previously done block
        # However it is wrong related to the full output row if the qk isn't the last block
        # And at this stage all the previously done qk_softmax blocks are also wrong and needs to be corrected
        # To correct previous blocks we will scale the accumulator
        # d_i / d_new is for correcting denominator
        # alpha is for correcting numerator
        acc_scale = d_i / d_new * alpha

        # This isn't useful in the algorithm, simply to fix a compiler bug
        # BUG: have to store and immediately load
        tl.store(t_ptrs, acc_scale)
        acc_scale = tl.load(t_ptrs)

        # acc scaling
        acc = acc * acc_scale[:, None]

        # We now apply the last operation, the multiplication by a block of matrix V
        v = tl.load(v_ptrs + n_row_offset * stride_vk)
        qk_softmax = qk_softmax.to(tl.float16)
        acc += tl.dot(qk_softmax, v)

        # We update the normalizer for the next iteration
        d_i = d_new
        l_i = l_new

    # For some reason we need to re-init this variable
    # The other variables in the original implementations seem not needed
    offs_n = tl.arange(0, BLOCK_DHEAD)
    off_o = head_idx * stride_oh + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    out_ptrs = output + off_o
    tl.store(out_ptrs, acc)


def attention_forward(q, k, v, sm_scale):
    """
    Computes attention

    @param q: Query matrix size (batch, heads, seq_length, dhead)
    @param k: Key matrix size (batch, heads, seq_length, dhead)
    @param v: Value matrix size (batch, heads, seq_length, dhead)
    @param sm_scale: Scaling factor applied after operation QxK
    @return:
    """
    # Constraints
    # Queries and keys have the same d_k size
    assert q.shape[-1] == k.shape[-1]
    batch, heads, seq_length, dhead = q.size()

    BLOCK_M = 128
    BLOCK_N = 128
    grid = (triton.cdiv(seq_length, BLOCK_M), batch * heads)
    tmp = torch.empty((batch * heads, seq_length), device=q.device, dtype=torch.float32)

    output = torch.empty_like(q)
    _fwd_kernel[grid](
        batch,
        heads,
        seq_length,
        q,
        k,
        v,
        sm_scale,
        tmp,
        output,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_DHEAD=dhead,
        num_warps=4,
        num_stages=1,
    )
    return output
