"""
Fused Attention
===============
This is a Triton implementation of the Flash Attention algorithm
(see: Dao et al., https://arxiv.org/pdf/2205.14135v2.pdf; Rabe and Staats https://arxiv.org/pdf/2112.05682v2.pdf)
"""

import pytest
import torch
import triton
import triton.language as tl


def generate_broadcast_mask(
    batch: int, heads: int, seq_length: int, dhead: int, dtype: torch.Tensor = torch.float32
) -> torch.Tensor:
    attention_mask = (
        torch.randint(1, seq_length, (batch,), device="cuda")[:, None]
        > torch.arange(0, seq_length, device="cuda")[None, :]
    )
    attention_mask = torch.reshape(attention_mask, (batch, 1, 1, seq_length))
    attention_mask = torch.where(attention_mask, 0, float("-inf"))
    attention_mask = attention_mask.to(dtype)
    # attention_mask = (1.0 - attention_mask) * torch.finfo(dtype).min
    return attention_mask


@triton.jit
def _fwd_kernel(
    Q,
    K,
    V,
    sm_scale,
    attention_mask_ptr,
    L,
    M,
    Out,
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qk,
    stride_kz,
    stride_kh,
    stride_kn,
    stride_kk,
    stride_vz,
    stride_vh,
    stride_vk,
    stride_vn,
    stride_oz,
    stride_oh,
    stride_om,
    stride_on,
    attention_mask_batch_stride,
    attention_mask_head_stride,
    attention_mask_m_stride,
    attention_mask_n_stride,
    Z,
    H,
    N_CTX,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    off_q = off_hz * stride_qh + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    off_k = off_hz * stride_qh + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk
    off_v = off_hz * stride_qh + offs_n[:, None] * stride_qm + offs_d[None, :] * stride_qk
    off_attention = off_hz * attention_mask_head_stride
    # Initialize pointers to Q, K, V
    q_ptrs = Q + off_q
    k_ptrs = K + off_k
    v_ptrs = V + off_v
    # initialize pointer to m and l
    m_prev = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_prev = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    # load q: it will stay in SRAM throughout
    q = tl.load(q_ptrs)
    # loop over k, v and update accumulator
    for start_n in range(0, N_CTX, BLOCK_N):
        # -- compute qk ----
        k = tl.load(k_ptrs)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, tl.trans(k))
        qk *= sm_scale
        # qk = tl.where(offs_m[:, None] >= (start_n + offs_n[None, :]), qk, float("-inf"))  # no causal mask
        # fake mask
        attention_mask_offs = off_attention + offs_n[None, :] * attention_mask_n_stride
        attention_mask = tl.load(
            attention_mask_ptr + attention_mask_offs,
            eviction_policy="evict_first",
        )
        qk = tl.where(attention_mask, qk, float("-inf"))  # no causal mask
        # compute new m
        m_curr = tl.maximum(tl.max(qk, 1), m_prev)
        # correct old l
        l_prev *= tl.exp(m_prev - m_curr)
        # attention weights
        p = tl.exp(qk - m_curr[:, None])
        l_curr = tl.sum(p, 1) + l_prev
        # rescale operands of matmuls
        l_rcp = 1.0 / l_curr
        p *= l_rcp
        acc *= (l_prev * l_rcp)[:, None]
        # update acc
        p = p.to(tl.float16)
        v = tl.load(v_ptrs)
        acc += tl.dot(p, v)
        # update m_i and l_i
        l_prev = l_curr
        m_prev = m_curr
        # update pointers
        k_ptrs += BLOCK_N * stride_kn
        v_ptrs += BLOCK_N * stride_vk
    # rematerialize offsets to save registers
    start_m = tl.program_id(0)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    # write back l and m
    l_ptrs = L + off_hz * N_CTX + offs_m
    m_ptrs = M + off_hz * N_CTX + offs_m
    tl.store(l_ptrs, l_prev)
    tl.store(m_ptrs, m_prev)
    # initialize pointers to output
    offs_n = tl.arange(0, BLOCK_DMODEL)
    off_o = off_hz * stride_oh + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    out_ptrs = Out + off_o
    tl.store(out_ptrs, acc)


class _attention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, sm_scale, attention_mask):
        BLOCK = 128
        # shape constraints
        Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
        assert Lq == Lk and Lk == Lv
        assert Lk in {16, 32, 64, 128}
        o = torch.empty_like(q)
        grid = (triton.cdiv(q.shape[2], BLOCK), q.shape[0] * q.shape[1], 1)
        L = torch.empty((q.shape[0] * q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
        m = torch.empty((q.shape[0] * q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
        num_warps = 4 if Lk <= 64 else 8

        _fwd_kernel[grid](
            q,
            k,
            v,
            sm_scale,
            attention_mask,
            L,
            m,
            o,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            k.stride(3),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            v.stride(3),
            o.stride(0),
            o.stride(1),
            o.stride(2),
            o.stride(3),
            *attention_mask.stride(),
            q.shape[0],
            q.shape[1],
            q.shape[2],
            BLOCK_M=BLOCK,
            BLOCK_N=BLOCK,
            BLOCK_DMODEL=Lk,
            num_warps=num_warps,
            num_stages=2,
        )
        # print(h.asm["ttgir"])

        ctx.save_for_backward(q, k, v, o, L, m)
        ctx.grid = grid
        ctx.sm_scale = sm_scale
        ctx.BLOCK_DMODEL = Lk
        return o


attention = _attention.apply


@pytest.mark.parametrize("Z, H, N_CTX, D_HEAD", [(4, 48, 128, 64), (4, 48, 512, 64)])
def test_op(Z, H, N_CTX, D_HEAD, dtype=torch.float16):
    torch.manual_seed(5)
    q = torch.rand((Z, H, N_CTX, D_HEAD), dtype=torch.float32, device="cuda")
    k = torch.rand((Z, H, N_CTX, D_HEAD), dtype=torch.float32, device="cuda")
    v = torch.rand((Z, H, N_CTX, D_HEAD), dtype=torch.float32, device="cuda")
    sm_scale = 0.3
    # reference implementation
    p = torch.matmul(q, k.transpose(2, 3)) * sm_scale
    # no causal mask
    # M = torch.tril(torch.ones((N_CTX, N_CTX), device="cuda"))
    # p = torch.where(M == 0, float("-inf"), p)
    # fake mask
    fake_attention_mask = generate_broadcast_mask(batch=Z, heads=H, seq_length=N_CTX, dhead=D_HEAD, dtype=torch.float32)

    p += fake_attention_mask
    p = torch.softmax(p.float(), dim=-1)#.half()
    ref_out = torch.matmul(p, v)

    # triton implementation
    tri_out = attention(q.half(), k.half(), v.half(), sm_scale, fake_attention_mask)
    # compare
    # print((ref_out - tri_out).abs().max().item())
    triton.testing.assert_almost_equal(ref_out.half(), tri_out.half(), decimal=1)
