import torch

batch, seq_len, heads, dim = 1, 16, 32, 128


def rms_norm_pytorch(x: torch.Tensor, rms_w: torch.Tensor, eps=1e-6) -> torch.Tensor:
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return x * rms_w


def reshape_for_broadcast_pytorch(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def rbe_pytorch(input_tensor: torch.Tensor, weights: torch.Tensor, freqs_cis: torch.Tensor):
    embenddings_rms_reshaped = input_tensor.view(-1, heads * dim)
    proj = torch.nn.functional.linear(embenddings_rms_reshaped, weights)
    proj_reshaped = proj.view(batch, -1, heads, dim)
    proj_reshaped = torch.view_as_complex(proj_reshaped.float().view(*proj_reshaped.shape[:-1], -1, 2))
    freqs_cis_reshaped = reshape_for_broadcast_pytorch(freqs_cis, proj_reshaped)
    out = torch.view_as_real(proj_reshaped * freqs_cis_reshaped).flatten(3)
    return out.type_as(input_tensor)


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1]), f"{freqs_cis.shape} != {(x.shape[1], x.shape[-1])}"
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def precompute_freqs_cis_pytorch(dim: int, end: int, theta: float = 10000.0):
    assert dim % 2 == 0

    # Generate a sequence of numbers from 0 to dim in steps of 2
    sequence = torch.arange(0, dim, 2, dtype=torch.float32, device="cuda")

    # Keep only the first half of the sequence (in case dim is odd?)
    # sequence = sequence[: (dim // 2)]

    # Calculate frequency values based on the sequence and theta
    freqs = 1.0 / (theta ** (sequence / dim))

    # Create a tensor of numbers from 0 to end, it represents the position ids
    t = torch.arange(end, device=freqs.device)

    # Generate a table of frequency values
    freqs = t[:, None] * freqs[None, :]  # torch.outer(t, freqs).float()

    # Calculate cosine and sine values for the frequencies
    # These can be considered as the real and imaginary parts of complex numbers
    freqs_cos = torch.cos(freqs)
    freqs_sin = torch.sin(freqs)

    # Return the cosine and sine values as two separate tensors
    return freqs_cos, freqs_sin


def apply_rotary_emb_pytorch(x: torch.Tensor, freq_cos: torch.Tensor, freq_sin: torch.Tensor) -> torch.Tensor:
    # Split x and x into real and imaginary parts
    x_real = x[..., 0::2]
    x_imag = x[..., 1::2]

    # Reshape freq_cos and freq_sin for broadcasting
    freq_cos = reshape_for_broadcast(freq_cos, x_real).to(torch.float32)
    freq_sin = reshape_for_broadcast(freq_sin, x_imag).to(torch.float32)

    # Perform the equivalent of complex multiplication
    x_out_real = x_real * freq_cos - x_imag * freq_sin
    x_out_imag = x_real * freq_sin + x_imag * freq_cos

    # Combine real and imaginary parts back into the original tensor
    x_out = torch.stack((x_out_real, x_out_imag), dim=-1).flatten(-2)

    return x_out.type_as(x)


def pytorch_all(input_tensor: torch.Tensor, weights: torch.Tensor, rms_weights: torch.Tensor, freqs_cis: torch.Tensor):
    embenddings_rms = rms_norm_pytorch(input_tensor, rms_weights)
    return rbe_pytorch(embenddings_rms, weights, freqs_cis)


def attention_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    output: torch.Tensor,
    sm_scale: float,
    is_causal: bool,
) -> torch.Tensor:
    p = torch.matmul(q, k.transpose(2, 3)) * sm_scale
    if is_causal:
        m_size = q.size(2)
        n_size = k.size(2)
        M = torch.tril(torch.ones((m_size, n_size), device="cuda"))
        p = torch.where(M == 0, float("-inf"), p)
    p = torch.nn.functional.softmax(p, dim=-1)
    ref_out = torch.matmul(p.to(v.dtype), v, out=output)
    return ref_out