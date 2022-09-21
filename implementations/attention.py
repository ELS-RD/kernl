import torch
import triton
import triton.language as tl


# CREDITS: Initially inspired by the Triton tutorial

@triton.jit
def _fwd_kernel(
        heads,
        seq_length,
        Q,
        K,
        V,
        sm_scale,
        mask,
        TMP,  # NOTE: TMP is a scratchpad buffer to workaround a compiler bug
        output,
        q_batch_stride, q_head_stride, q_m_stride, q_k_stride,
        k_batch_stride, k_head_stride, k_n_stride, k_k_stride, # n et k inversé ici (naming only) ???????????????????????????????????
        v_batch_stride, v_head_stride, v_k_stride, v_n_stride,
        o_batch_stride, o_head_stride, o_m_stride, o_n_stride,
        mask_batch_stride, mask_head_stride, mask_m_stride, mask_k_stride,
        IS_MASK_BROADCAST: tl.constexpr,
        IS_CAUSAL: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_DHEAD: tl.constexpr,
        BLOCK_N: tl.constexpr,
):
    """
    Computes attention
    Each head is size (seq_length, BLOCK_DHEAD)

    @param heads: Number of heads per batch
    @param seq_length: Sequence length
    @param Q: Query matrix size (batch, heads, seq_length, BLOCK_DHEAD)
    @param K: Key matrix size (batch, heads, seq_length, BLOCK_DHEAD)
    @param V: Value matrix size (batch, heads, seq_length, BLOCK_DHEAD)
    @param sm_scale: Scaling factor applied after operation QxK
    @param TMP: Temporary variable to fix a compiler bug
    @param output: Output matrix size (batch, heads, seq_length, BLOCK_DHEAD)
    @param q_batch_stride: matrix q stride for batch dimension
    @param q_head_stride: matrix q stride for head dimension
    @param q_m_stride: matrix q stride for rows, called "dimension m"
    @param q_k_stride: matrix q stride for columns
    @param k_batch_stride: matrix k stride for batch dimension
    @param k_head_stride: matrix k stride for head dimension
    @param k_n_stride: matrix k stride for rows, called "dimension n"
    @param k_k_stride: matrix k stride for columns
    @param v_batch_stride: matrix v stride for batch dimension
    @param v_head_stride: matrix v stride for head dimension
    @param v_k_stride: matrix v stride for columns
    @param v_n_stride: matrix v stride for rows
    @param o_batch_stride: output matrix stride for batch dimension
    @param o_head_stride: output matrix stride for head dimension
    @param o_m_stride: output matrix stride for rows
    @param o_n_stride: output matrix stride for columns
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

    current_batch_idx = head_idx // heads
    current_head_idx = head_idx % heads
    # memory offsets matrices on whole Q, K, V matrices
    # Offsets for the current block on matrix Q
    off_q = current_batch_idx * q_batch_stride \
            + current_head_idx * q_head_stride \
            + offs_m[:, None] * q_m_stride + offs_d[None, :] * q_k_stride

    # Offsets for the first block on matrix K
    off_k = current_batch_idx * k_batch_stride \
            + current_head_idx * k_head_stride + offs_n[:, None] * k_n_stride \
            + offs_d[None, :] * k_k_stride

    # Offsets for the first block on matrix V
    off_v = current_batch_idx * v_batch_stride \
            + current_head_idx * v_head_stride + offs_n[:, None] * q_m_stride \
            + offs_d[None, :] * q_k_stride

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

    n_end = seq_length
    if IS_CAUSAL:
        n_end = (m_block_idx + 1) * BLOCK_M,

    # loop over k, v and update accumulator
    # n_row_offset is the row offset on dimension N of the current block
    # It's used for booth the N dimension of K and V because they are handled at the same time
    for n_row_offset in range(0, n_end, BLOCK_N):
        n_row_offset = tl.multiple_of(n_row_offset, BLOCK_N)
        # We load the current block in K in SRAM
        # We do the first multiplication between the block in Q and the current block in K
        # We finish with the scaling (sqrt(BLOCK_DHEAD) in Vaswani et al. but sm_scale here)
        k = tl.load(k_ptrs + n_row_offset * k_n_stride)
        qk = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        qk += tl.dot(q, k, trans_b=True)
        qk *= sm_scale

        if IS_CAUSAL:
            qk += tl.where(offs_m[:, None] >= (n_row_offset + offs_n[None, :]), 0, float("-inf"))

        # Todo: Try to extract outside of this loop to see if perf improvement
        if mask is not None:
            offs_mask = current_batch_idx * mask_batch_stride \
                        + (offs_n[None, :] + n_row_offset) * mask_k_stride
            if not IS_MASK_BROADCAST:
                offs_mask += offs_m[:, None] * mask_m_stride \
                             + current_head_idx * mask_head_stride
            mask_ptrs = mask + offs_mask
            m = tl.load(mask_ptrs)
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

        # We correct the numerator for the current softmax (*beta) -> exp(l_j - l_new) * exp(qk - mj) = exp(l_new) We
        # divide by the normalization. It's strange to do it this way instead of simply computing the softmax for qk,
        # but since all needed operations are already done for updating m and d, it seems faster
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
        tl.store(t_ptrs, acc_scale)
        acc_scale = tl.load(t_ptrs)

        # acc scaling
        acc = acc * acc_scale[:, None]

        # We now apply the last operation, the multiplication by a block of matrix V
        v = tl.load(v_ptrs + n_row_offset * v_k_stride)
        qk_softmax = qk_softmax.to(tl.float16)
        acc += tl.dot(qk_softmax, v)

        # We update the normalizer for the next iteration
        d_i = d_new
        l_i = l_new

    # For some reason we need to re-init this variable
    # The other variables in the original implementations seem not needed
    offs_n = tl.arange(0, BLOCK_DHEAD)
    off_o = current_batch_idx * o_batch_stride \
            + current_head_idx * o_head_stride + offs_m[:, None] * o_m_stride \
            + offs_n[None, :] * o_n_stride

    out_ptrs = output + off_o
    tl.store(out_ptrs, acc)


def attention_reference(q, k, v, sm_scale, attention_mask=None):
    """
    Reference implementation for attention
    @param q: Query matrix size (batch, heads, seq_length, BLOCK_DHEAD)
    @param k: Key matrix size (batch, heads, seq_length, BLOCK_DHEAD)
    @param v: Value matrix size (batch, heads, seq_length, BLOCK_DHEAD)
    @param sm_scale: Scaling factor applied after operation QxK
    @param attention_mask: Size (batch, 1, 1, seq_length) or (batch, heads, seq_length, seq_length). Warning the mask isn't a binary mask
    like the one you use normally. This mask is directly added to QxK.
    @return:
    """
    p = torch.matmul(q, k.transpose(2, 3)) * sm_scale

    if attention_mask is not None:
        p += attention_mask
    p = torch.nn.functional.softmax(p, dim=-1)
    ref_out = torch.matmul(p, v)
    return ref_out


def attention_forward(q, k, v, output, sm_scale, is_causal=False, mask=None):
    """
    Computes attention

    @param q: Query matrix size (batch, heads, seq_length, dhead)
    @param k: Key matrix size (batch, heads, seq_length, dhead)
    @param v: Value matrix size (batch, heads, seq_length, dhead)
    @param output: Output matrix size (batch, heads, seq_length, dhead)
    @param sm_scale: Scaling factor applied after operation QxK
    @param is_causal: Autoregressive decoder attention
    @return:
    """
    # Constraints
    # Queries and keys have the same d_k size
    assert q.shape[-1] == k.shape[-1]
    batch, heads, seq_length, dhead = q.size()

    # Todo: Ensure 2^n only ?
    BLOCK_M = min(128, seq_length)
    BLOCK_N = min(128, seq_length)
    assert seq_length % BLOCK_M == seq_length % BLOCK_N == 0

    grid = (triton.cdiv(seq_length, BLOCK_M), batch * heads)
    tmp = torch.empty((batch * heads, seq_length), device=q.device, dtype=torch.float32)

    if mask is not None:
        assert mask.size() == (batch, heads, seq_length, seq_length) or mask.size(1) == 1 and mask.size(2) == 1

    _fwd_kernel[grid](
        heads,
        seq_length,
        q,
        k,
        v,
        sm_scale,
        mask,
        tmp,
        output,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        mask.stride(0) if mask is not None else 0,
        mask.stride(1) if mask is not None else 0,
        mask.stride(2) if mask is not None else 0,
        mask.stride(3) if mask is not None else 0,
        IS_MASK_BROADCAST=mask.size() != (batch, heads, seq_length, seq_length) if mask is not None else False,
        IS_CAUSAL=is_causal,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_DHEAD=dhead,
        num_warps=4,
        num_stages=1,
    )
    return output