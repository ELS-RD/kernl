import torch

import triton
import triton.language as tl

from kernel.pytorch_reference import rms_norm_pytorch, precompute_freqs_cis_pytorch, apply_rotary_emb_pytorch

torch.manual_seed(1234)


@triton.jit
def rmsnorm_triton(x_ptr, rms_w_ptr, output_ptr,
                   stride_x_batch, stride_x_m, stride_x_k,
                   stride_rms_w,
                   stride_out_batch, stride_out_m, stride_out_k,
                   N_SIZE: tl.constexpr, eps: tl.constexpr, BLOCK_N_SIZE: tl.constexpr):
    pid_batch = tl.program_id(0)
    pid_m = tl.program_id(1)

    offs_m = pid_batch * stride_x_batch + pid_m * stride_x_m
    block_N = tl.arange(0, BLOCK_N_SIZE)
    var = tl.zeros((BLOCK_N_SIZE,), tl.float32)
    for block_n_start_idx in range(0, N_SIZE, BLOCK_N_SIZE):
        offs_n = block_n_start_idx + block_N
        x_ptr_mask = offs_n < N_SIZE
        x = tl.load(x_ptr + offs_m + offs_n * stride_x_k, mask=x_ptr_mask, other=0.0)
        var += tl.math.pow(x.to(tl.float32), 2)

    var = tl.sum(var, axis=0) / N_SIZE
    rstd = tl.math.rsqrt(var + eps)

    # multiply by weight and add bias
    for block_n_start_idx in range(0, N_SIZE, BLOCK_N_SIZE):
        offs_n = block_n_start_idx + block_N
        x_ptr_mask = offs_n < N_SIZE
        rms_w = tl.load(rms_w_ptr + offs_n * stride_rms_w, mask=x_ptr_mask)

        x = tl.load(x_ptr + offs_m + offs_n * stride_x_k, mask=x_ptr_mask, other=0.0).to(tl.float32)
        x_hat = x * rstd
        out = x_hat * rms_w
        out_off = pid_batch * stride_out_batch + pid_m * stride_out_m + offs_n * stride_out_k
        tl.store(output_ptr + out_off, out, mask=x_ptr_mask)


def rmsnorm_triton_wrapper(x, rms_w, eps=1e-6):
    batch, M, K = x.shape
    assert rms_w.shape[-1] == K
    out = torch.empty_like(x)
    rmsnorm_triton[(batch, M,)](x, rms_w, out,
                                *x.stride(),
                                *rms_w.stride(),
                                *out.stride(),
                                N_SIZE=K, eps=eps, BLOCK_N_SIZE=1024,
                                )
    return out


@triton.jit
def get_freq_multi_tokens(offs_cn, starting_idx, theta: tl.constexpr, NB_TOKENS: tl.constexpr):
    DIM: tl.constexpr = 128  # in model, dim = self.params.dim // self.params.n_heads
    freqs = offs_cn % DIM
    freqs = freqs.to(tl.float32) / DIM
    freqs = tl.math.pow(theta, freqs)
    freqs = (tl.arange(0, NB_TOKENS) + starting_idx)[:, None] / freqs[None, :]
    return tl.cos(freqs), tl.sin(freqs)


@triton.jit
def rbe_triton(x_ptr, out_ptr,
               M, K,
               stride_x_batch, stride_x_m, stride_x_n,
               stride_out_batch, stride_out_m, stride_out_n,
               start_token_position,
               THETA: tl.constexpr, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_K: tl.constexpr):
    pid_batch = tl.program_id(axis=0)
    pid = tl.program_id(axis=1)
    pid_m = pid // tl.cdiv(K, BLOCK_SIZE_K)
    pid_n = pid % tl.cdiv(K, BLOCK_SIZE_K)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K // 2) * 2  # take only even numbers
    x_ptrs = x_ptr + (pid_batch * stride_x_batch + stride_x_m * offs_m[:, None] + stride_x_n * offs_n[None, :])
    x_real_mask = (offs_m[:, None] < M) & (offs_n[None, :] < K)
    real = tl.load(x_ptrs, mask=x_real_mask, other=0.0)
    x_imag_mask = (offs_m[:, None] < M) & (1 + offs_n[None, :] < K)
    imag = tl.load(x_ptrs + 1, mask=x_imag_mask, other=0.0)
    tl.debug_barrier()
    start_block = start_token_position + pid_m * BLOCK_SIZE_M
    cos, sin = get_freq_multi_tokens(offs_cn=offs_n, starting_idx=start_block, theta=THETA, NB_TOKENS=BLOCK_SIZE_M)

    out_real = real * cos - imag * sin
    out_imag = real * sin + imag * cos
    tl.debug_barrier()
    out_ptrs = out_ptr + (
            pid_batch * stride_out_batch + stride_out_m * offs_m[:, None] + stride_out_n * offs_n[None, :])
    out_real_mask = (offs_m[:, None] < M) & (offs_n[None, :] < K)
    tl.store(out_ptrs, out_real, mask=out_real_mask)
    out_imag_mask = (offs_m[:, None] < M) & (1 + offs_n[None, :] < K)
    tl.store(out_ptrs + 1, out_imag, mask=out_imag_mask)


def rbe_triton_wrapper(x: torch.Tensor, pos: int) -> torch.Tensor:
    batch, M, K = x.shape
    out = torch.empty_like(x)
    grid = lambda META: (
        batch, triton.cdiv(META["M"], META["BLOCK_SIZE_M"]) * triton.cdiv(META["K"], META["BLOCK_SIZE_K"]),)

    rbe_triton[grid](x, out,
                     M, K,
                     *x.stride(),
                     *out.stride(),
                     start_token_position=pos, THETA=10000., BLOCK_SIZE_M=2, BLOCK_SIZE_K=1024)
    return out


@triton.jit
def rms_matmul_rbe(
        x_ptr, w_ptr, rms_w_ptr, out_ptr,
        M, N, K,
        stride_x_batch, stride_x_m, stride_x_k,
        stride_w_k, stride_w_n,
        stride_rms_w,
        stride_out_batch, stride_out_m, stride_out_n,
        start_token_position,
        USE_FP8: tl.constexpr,
        RBE_EPILOGUE: tl.constexpr,
        THETA: tl.constexpr,
        EPS: tl.constexpr,
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    """
    Prologue: RMS
    Epilogue: nothing or Rotary embeddings
    c = ROBE((rms(a) * rms_w) @ b)
    """
    pid_batch = tl.program_id(axis=0)
    pid = tl.program_id(axis=1)
    pid_m = pid // tl.cdiv(N, BLOCK_SIZE_N)
    pid_n = pid % tl.cdiv(N, BLOCK_SIZE_N)

    offs_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    x_ptrs = x_ptr + (pid_batch * stride_x_batch + offs_m[:, None] * stride_x_m + offs_k[None, :] * stride_x_k)
    w_ptrs = w_ptr + (offs_k[:, None] * stride_w_k + offs_n[None, :] * stride_w_n)
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    rms_w_ptrs = rms_w_ptr + tl.arange(0, BLOCK_SIZE_K)[None, :] * stride_rms_w
    x_sum = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)
    for _ in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        x = tl.load(x_ptrs)
        x_sum += tl.math.pow(x.to(tl.float32), 2)
        rms_w = tl.load(rms_w_ptrs)  # TODO add an assert that rms_w is a multiple of BLOCK SIZE K
        if USE_FP8:
            rms_w = rms_w.to(tl.float8e5, bitcast=True)
            rms_w = rms_w.to(tl.float16)
        x = x * rms_w
        w = tl.load(w_ptrs)  # TODO add an assert that w is a multiple of BLOCK SIZE K
        if USE_FP8:
            w = w.to(tl.float8e5, bitcast=True)
            w = w.to(tl.float32)
            w = w.to(tl.float16)
        accumulator += tl.dot(x, w)
        x_ptrs += BLOCK_SIZE_K * stride_x_k
        w_ptrs += BLOCK_SIZE_K * stride_w_k
        rms_w_ptrs += BLOCK_SIZE_K * stride_rms_w
    x_mean = tl.sum(x_sum, axis=1) / K + EPS
    x_norm = tl.math.rsqrt(x_mean)
    accumulator = accumulator * x_norm[:, None]

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    out_ptrs = out_ptr + (
                pid_batch * stride_out_batch + offs_m[:, None] * stride_out_m + offs_n[None, :] * stride_out_n)
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    if RBE_EPILOGUE:
        tl.store(out_ptrs, accumulator, mask=out_mask)
        tl.debug_barrier()
        rbe_triton(out_ptr, out_ptr, M, N, stride_out_batch, stride_out_m, stride_out_n, stride_out_batch, stride_out_m,
                   stride_out_n, start_token_position, THETA,
                   BLOCK_SIZE_M, BLOCK_SIZE_N)
    else:
        tl.store(out_ptrs, accumulator, mask=out_mask)


def rms_matmul_rbe_wrapper(x: torch.Tensor, weight: torch.Tensor, rms_w: torch.Tensor, use_rbe: bool, start_pos: int,
                           n_heads: int, head_dim: int):
    assert weight.dtype == rms_w.dtype
    assert weight.dtype in [torch.float16, torch.int8]
    batch, M, K = x.shape
    weight_t = weight.t()
    K_W, N = weight_t.shape
    assert K == K_W
    out = torch.empty((batch, M, N), dtype=weight_t.dtype, device=weight_t.device)  # TODO replace by empty
    out_ptr = triton.reinterpret(out, tl.float8e5 if out.dtype == torch.int8 else tl.float16)

    grid = lambda META: (
    batch, triton.cdiv(META["M"], META["BLOCK_SIZE_M"]) * triton.cdiv(META["N"], META["BLOCK_SIZE_N"]))

    rms_matmul_rbe[grid](
        x_ptr=x,
        w_ptr=weight_t, rms_w_ptr=rms_w, out_ptr=out_ptr,
        M=M, N=N, K=K,
        stride_x_batch=x.stride(0), stride_x_m=x.stride(1), stride_x_k=x.stride(2),
        stride_w_k=weight_t.stride(0), stride_w_n=weight_t.stride(1),
        stride_rms_w=rms_w.stride(0),
        stride_out_batch=out.stride(0), stride_out_m=out.stride(1), stride_out_n=out.stride(2),
        start_token_position=start_pos,
        USE_FP8=weight_t.dtype == torch.int8,
        RBE_EPILOGUE=use_rbe,
        THETA=10000.,
        EPS=1e-6,
        BLOCK_SIZE_M=16, BLOCK_SIZE_N=64, BLOCK_SIZE_K=64,
        num_stages=4, num_warps=4
    )
    out = out.view(batch, M, n_heads, head_dim)
    return out


@triton.jit
def rms_matmul_rbe_qkv(x_ptr,
                       q_weight_ptr, k_weight_ptr, v_weight_ptr,
                       rms_w_ptr,
                       q_ptr, k_ptr, v_ptr,
                       M, N, K,
                       stride_x_batch, stride_x_m, stride_x_k,
                       stride_q_w_k, stride_q_w_n,
                       stride_k_w_k, stride_k_w_n,
                       stride_v_w_k, stride_v_w_n,
                       stride_rms_w,
                       stride_q_batch, stride_q_m, stride_q_n,
                       stride_k_batch, stride_k_m, stride_k_n,
                       stride_v_batch, stride_v_m, stride_v_n,
                       start_token_position,
                       USE_FP8: tl.constexpr,
                       THETA: tl.constexpr,
                       EPS: tl.constexpr,
                       BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr):
    # q
    rms_matmul_rbe(
        x_ptr=x_ptr,
        w_ptr=q_weight_ptr, rms_w_ptr=rms_w_ptr, out_ptr=q_ptr,
        M=M, N=N, K=K,
        stride_x_batch=stride_x_batch, stride_x_m=stride_x_m, stride_x_k=stride_x_k,
        stride_w_k=stride_q_w_k, stride_w_n=stride_q_w_n,
        stride_rms_w=stride_rms_w,
        stride_out_batch=stride_q_batch, stride_out_m=stride_q_m, stride_out_n=stride_q_n,
        start_token_position=start_token_position,
        USE_FP8=USE_FP8,
        RBE_EPILOGUE=True,
        THETA=THETA,
        EPS=EPS,
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    # k
    rms_matmul_rbe(
        x_ptr=x_ptr,
        w_ptr=k_weight_ptr, rms_w_ptr=rms_w_ptr, out_ptr=k_ptr,
        M=M, N=N, K=K,
        stride_x_batch=stride_x_batch, stride_x_m=stride_x_m, stride_x_k=stride_x_k,
        stride_w_k=stride_k_w_k, stride_w_n=stride_k_w_n,
        stride_rms_w=stride_rms_w,
        stride_out_batch=stride_k_batch, stride_out_m=stride_k_m, stride_out_n=stride_k_n,
        start_token_position=start_token_position,
        USE_FP8=USE_FP8,
        RBE_EPILOGUE=True,
        THETA=THETA,
        EPS=EPS,
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    # v
    rms_matmul_rbe(
        x_ptr=x_ptr,
        w_ptr=v_weight_ptr, rms_w_ptr=rms_w_ptr, out_ptr=v_ptr,
        M=M, N=N, K=K,
        stride_x_batch=stride_x_batch, stride_x_m=stride_x_m, stride_x_k=stride_x_k,
        stride_w_k=stride_v_w_k, stride_w_n=stride_v_w_n,
        stride_rms_w=stride_rms_w,
        stride_out_batch=stride_v_batch, stride_out_m=stride_v_m, stride_out_n=stride_v_n,
        start_token_position=start_token_position,
        USE_FP8=USE_FP8,
        RBE_EPILOGUE=False,
        THETA=THETA,
        EPS=EPS,
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K,
    )


def rms_matmul_rbe_qkv_wrapper(x: torch.Tensor,
                               start_pos: int,
                               q_weight: torch.Tensor, k_weight: torch.Tensor, v_weight: torch.Tensor,
                               rms_w: torch.Tensor,
                               n_heads: int, head_dim: int,
                               k: torch.Tensor,
                               v: torch.Tensor,
                               eps: float = 1e-6, theta=10000.):
    assert q_weight.shape == k_weight.shape == v_weight.shape
    assert q_weight.dtype == k_weight.dtype == v_weight.dtype == rms_w.dtype
    assert q_weight.dtype in [torch.float16, torch.int8]
    batch, M, K = x.shape

    assert K == rms_w.shape[0]

    q_weight_t = q_weight.t()
    k_weight_t = k_weight.t()
    v_weight_t = v_weight.t()
    K_W, N = q_weight_t.shape
    assert K == K_W
    q = torch.empty((batch, M, N), dtype=torch.float16, device=q_weight_t.device)

    k = k.view((batch, M, N))
    v = v.view((batch, M, N))
    assert k.dtype == k_weight.dtype
    assert v.dtype == v_weight.dtype

    q_ptr = triton.reinterpret(q, tl.float16)
    k_ptr = triton.reinterpret(k, tl.float8e5 if k.dtype == torch.int8 else tl.float16)
    v_ptr = triton.reinterpret(v, tl.float8e5 if v.dtype == torch.int8 else tl.float16)

    grid = lambda META: (
    batch, triton.cdiv(META["M"], META["BLOCK_SIZE_M"]) * triton.cdiv(META["N"], META["BLOCK_SIZE_N"]))

    rms_matmul_rbe_qkv[grid](
        x_ptr=x,
        q_weight_ptr=q_weight_t, k_weight_ptr=k_weight_t, v_weight_ptr=v_weight_t,
        rms_w_ptr=rms_w,
        q_ptr=q_ptr, k_ptr=k_ptr, v_ptr=v_ptr,
        M=M, N=N, K=K,
        stride_x_batch=x.stride(0), stride_x_m=x.stride(1), stride_x_k=x.stride(2),
        stride_q_w_k=q_weight_t.stride(0), stride_q_w_n=q_weight_t.stride(1),
        stride_k_w_k=k_weight_t.stride(0), stride_k_w_n=k_weight_t.stride(1),
        stride_v_w_k=v_weight_t.stride(0), stride_v_w_n=v_weight_t.stride(1),
        stride_rms_w=rms_w.stride(0),
        stride_q_batch=q.stride(0), stride_q_m=q.stride(1), stride_q_n=q.stride(2),
        stride_k_batch=k.stride(0), stride_k_m=k.stride(1), stride_k_n=k.stride(2),
        stride_v_batch=v.stride(0), stride_v_m=v.stride(1), stride_v_n=v.stride(2),
        start_token_position=start_pos,
        USE_FP8=q_weight.dtype == torch.int8,
        THETA=theta,
        EPS=eps,
        BLOCK_SIZE_M=16, BLOCK_SIZE_N=64, BLOCK_SIZE_K=64,
        num_stages=4, num_warps=4
    )
    q = q.view(batch, M, n_heads, head_dim)
    k = k.view(batch, M, n_heads, head_dim)
    v = v.view(batch, M, n_heads, head_dim)
    return q, k, v


batch, seq_len, heads, dim = [1, 16, 32, 128]

embeddings_load = torch.randn([batch, seq_len, heads * dim], dtype=torch.float16, device="cuda")
rms_weights = torch.randn([heads * dim], dtype=torch.float16, device="cuda") * 0.2
q_weights_load = torch.randn([heads * dim, heads * dim], dtype=torch.float16, device="cuda") * 0.2


out_rms_triton = rmsnorm_triton_wrapper(x=embeddings_load, rms_w=rms_weights)
out_rms_pytorch = rms_norm_pytorch(x=embeddings_load, rms_w=rms_weights)
assert torch.allclose(out_rms_triton, out_rms_pytorch, atol=1e-1)
print("rmsnorm triton", triton.testing.do_bench(lambda: rmsnorm_triton_wrapper(x=embeddings_load, rms_w=rms_weights)))
print("rmsnorm pytorch", triton.testing.do_bench(lambda: rms_norm_pytorch(x=embeddings_load, rms_w=rms_weights)))


xq = out_rms_pytorch @ q_weights_load.t()
xq = xq.view(batch, seq_len, heads, dim)


xq_output_triton = out_rms_triton @ q_weights_load.t()
out_rbe_triton = rbe_triton_wrapper(xq_output_triton, pos=0).view(batch, seq_len, heads, dim)

freq_cos, freq_sin = precompute_freqs_cis_pytorch(dim=128, end=seq_len)
out_rbe_pytorch = apply_rotary_emb_pytorch(x=xq, freq_cos=freq_cos, freq_sin=freq_sin).view(batch, seq_len, heads, dim)
assert torch.allclose(out_rbe_pytorch, out_rbe_triton, atol=1e-1), f"max diff: {torch.max(torch.abs(out_rbe_pytorch - out_rbe_triton))}"
print("rbe triton", triton.testing.do_bench(lambda: rbe_triton_wrapper(xq_output_triton, pos=0)))
print("rbe pytorch", triton.testing.do_bench(lambda: apply_rotary_emb_pytorch(x=xq, freq_cos=freq_cos, freq_sin=freq_sin)))


out_rms_matmul_rbe_triton = rms_matmul_rbe_wrapper(x=embeddings_load, start_pos=0, weight=q_weights_load, rms_w=rms_weights,
                                                   use_rbe=True, n_heads=32, head_dim=128).view(batch, seq_len, heads, dim)
assert torch.allclose(out_rms_matmul_rbe_triton, out_rbe_pytorch, atol=1e-1)


def get_rms_matmul_rbe_pytorch():
    a = rms_norm_pytorch(x=embeddings_load, rms_w=rms_weights)
    b = a @ q_weights_load.t()
    b = b.view(batch, seq_len, heads, dim)
    apply_rotary_emb_pytorch(x=b, freq_cos=freq_cos, freq_sin=freq_sin)


print("rms_matmul_rbe triton", triton.testing.do_bench(lambda: rms_matmul_rbe_wrapper(x=embeddings_load, start_pos=0, weight=q_weights_load, rms_w=rms_weights,
                                                                                      use_rbe=True, n_heads=32, head_dim=128)))
print("rms_matmul_rbe pytorch", triton.testing.do_bench(get_rms_matmul_rbe_pytorch))

k = torch.empty((embeddings_load.shape[0], embeddings_load.shape[1], q_weights_load.shape[-1]),
                dtype=q_weights_load.dtype, device=q_weights_load.device)
v = torch.empty_like(k)
out_rms_matmul_rbe_qkv, _, _ = rms_matmul_rbe_qkv_wrapper(x=embeddings_load, start_pos=0,
                                                          q_weight=q_weights_load, k_weight=q_weights_load,
                                                          v_weight=q_weights_load, rms_w=rms_weights,
                                                          k=k, v=v,
                                                          n_heads=32,
                                                          head_dim=128)

assert torch.allclose(out_rms_matmul_rbe_qkv, out_rbe_pytorch, atol=1e-1)

position = 5
embeddings_load_1_token = embeddings_load[:, position:position + 1, :]
xq_cplx_rotated_loaded_1_token = out_rbe_pytorch[:, position:position + 1, ...]
k_1_token = k[:, position:position + 1, :]
v_1_token = v[:, position:position + 1, :]

_, out_rms_matmul_rbe_qkv_1_token, _ = rms_matmul_rbe_qkv_wrapper(x=embeddings_load_1_token, start_pos=position,
                                                                  q_weight=q_weights_load, k_weight=q_weights_load,
                                                                  v_weight=q_weights_load, rms_w=rms_weights,
                                                                  k=k_1_token, v=v_1_token,
                                                                  n_heads=32,
                                                                  head_dim=128)
assert torch.allclose(out_rms_matmul_rbe_qkv_1_token, xq_cplx_rotated_loaded_1_token, atol=1e-1)


def get_qkv_rms_matmul_rbe_pytorch():
    for i in range(3):
        a = rms_norm_pytorch(x=embeddings_load, rms_w=rms_weights)
        b = a @ q_weights_load.t()
        b = b.view(batch, seq_len, heads, dim)
        if i == 2:
            apply_rotary_emb_pytorch(x=b, freq_cos=freq_cos, freq_sin=freq_sin)

print("qkv rms_matmul_rbe triton", triton.testing.do_bench(lambda: rms_matmul_rbe_qkv_wrapper(x=embeddings_load, start_pos=0,
                                                                                              q_weight=q_weights_load, k_weight=q_weights_load,
                                                                                              v_weight=q_weights_load, rms_w=rms_weights,
                                                                                              k=k, v=v,
                                                                                              n_heads=32,
                                                                                              head_dim=128)))
print("qkv rms_matmul_rbe pytorch", triton.testing.do_bench(get_qkv_rms_matmul_rbe_pytorch))