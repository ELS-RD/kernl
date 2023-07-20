# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# code commented below is to replace parallel exec by simpler local exec
# it is not strictly required but remove some small PyTorch overhead
# plus it makes code simpler to launch and think about

import math
from dataclasses import dataclass
from typing import Any, Optional, Tuple

# import fairscale.nn.model_parallel.initialize as fs_init
import torch
import torch.nn.functional as F
# from fairscale.nn.model_parallel.layers import (
#     ColumnParallelLinear,
#     ParallelEmbedding,
#     RowParallelLinear,
# )
from torch import nn
from utils.config import Config
from utils.nvtx_fake import NoOpContextManager

config = Config()


try:
    import nvtx
except ImportError:
    config.set_nvtx(False)


@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1  # defined later by tokenizer
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5

    max_batch_size: int = 32
    max_seq_len: int = 2048


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
        xq: torch.Tensor,
        xk: torch.Tensor,
        freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        model_parallel_size = 1  # fs_init.get_model_parallel_world_size()
        self.n_local_heads = args.n_heads // model_parallel_size
        self.n_local_kv_heads = self.n_kv_heads // model_parallel_size
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads

        self.wq = torch.nn.Linear(  # ColumnParallelLinear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
            # gather_output=False,
            # init_method=lambda x: x,
        )
        self.wk = torch.nn.Linear(  # ColumnParallelLinear(
            args.dim,
            self.n_kv_heads * self.head_dim,
            bias=False,
            # gather_output=False,
            # init_method=lambda x: x,
        )
        self.wv = torch.nn.Linear(  # ColumnParallelLinear(
            args.dim,
            self.n_kv_heads * self.head_dim,
            bias=False,
            # gather_output=False,
            # init_method=lambda x: x,
        )
        self.wo = torch.nn.Linear(  # RowParallelLinear(
            args.n_heads * self.head_dim,
            args.dim,
            bias=False,
            # input_is_parallel=True,
            # init_method=lambda x: x,
        )

        self.cache_k = torch.zeros(
            (
                args.max_batch_size,
                args.max_seq_len,
                self.n_local_kv_heads,
                self.head_dim,
            )
        ).cuda()
        self.cache_v = torch.zeros(
            (
                args.max_batch_size,
                args.max_seq_len,
                self.n_local_kv_heads,
                self.head_dim,
            )
        ).cuda()

    def forward(
            self,
            x: torch.Tensor,
            x_norm: Optional[torch.Tensor],
            start_pos: int,
            freqs_cis: torch.Tensor,
            mask: Optional[torch.Tensor],
            rms_weights: torch.Tensor
    ):
        bsz, seqlen, _ = x.shape

        if config.get_use_triton():
            from kernel.fused_kernel_proj_qkv import rms_matmul_rbe_qkv_wrapper
            with nvtx.annotate(message=f"RMS projection RBE (element W + matmul)", color="red"):
                cache_k = self.cache_k[:bsz, start_pos: start_pos + seqlen]
                cache_v = self.cache_v[:bsz, start_pos: start_pos + seqlen]
                xq, xk, xv = rms_matmul_rbe_qkv_wrapper(x=x, start_pos=start_pos, q_weight=self.wq.weight,
                                                        k_weight=self.wk.weight, v_weight=self.wv.weight,
                                                        rms_w=rms_weights, n_heads=self.n_local_heads,
                                                        head_dim=self.head_dim, k=cache_k, v=cache_v)
        else:
            with nvtx.annotate(message=f"QKV proj", color="red"):
                xq, xk, xv = self.wq(x_norm), self.wk(x_norm), self.wv(x_norm)

            xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
            xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
            xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
            with nvtx.annotate(message=f"RBE (element W)", color="blue"):
                xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

            self.cache_k = self.cache_k.to(xq)
            self.cache_v = self.cache_v.to(xq)

            self.cache_k[:bsz, start_pos: start_pos + seqlen] = xk
            self.cache_v[:bsz, start_pos: start_pos + seqlen] = xv

        keys = self.cache_k[:bsz, : start_pos + seqlen]
        values = self.cache_v[:bsz, : start_pos + seqlen]

        # repeat k/v heads if n_kv_heads < n_heads
        keys = repeat_kv(keys, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)
        values = repeat_kv(values, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)

        xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        with nvtx.annotate(message=f"attention score computation", color="red"):
            scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask  # (bs, n_local_heads, seqlen, cache_len + seqlen)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        with nvtx.annotate(message=f"attention score application", color="red"):
            output = torch.matmul(scores, values)  # (bs, n_local_heads, seqlen, head_dim)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        with nvtx.annotate(message=f"output projection", color="red"):
            output = self.wo(output)
        return output


class FeedForward(nn.Module):
    def __init__(
            self,
            dim: int,
            hidden_dim: int,
            multiple_of: int,
            ffn_dim_multiplier: Optional[float],
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = torch.nn.Linear(  # ColumnParallelLinear(
            dim, hidden_dim, bias=False,  # gather_output=False, init_method=lambda x: x
        )
        self.w2 = torch.nn.Linear(  # RowParallelLinear(
            hidden_dim, dim, bias=False,  # input_is_parallel=True, init_method=lambda x: x
        )
        self.w3 = torch.nn.Linear(  # ColumnParallelLinear(
            dim, hidden_dim, bias=False,  # gather_output=False, init_method=lambda x: x
        )

    def forward(self, x, x_norm, rms_weights):
        if config.get_use_triton():
            from kernel.fused_kernel_ff import kernel_ff
            with nvtx.annotate(message=f"FF1 (matmul)", color="red"):
                silu_times_w3 = kernel_ff(x, self.w1.weight, self.w3.weight, rms_weights)
                w2_out = self.w2(silu_times_w3)
                return w2_out
        else:
            with nvtx.annotate(message=f"FFN (W1)", color="red"):
                w1_out = self.w1(x_norm)
            silu_out = F.silu(w1_out)
            with nvtx.annotate(message=f"FFN (W3)", color="red"):
                w3_out = self.w3(x_norm)
            silu_times_w3 = silu_out * w3_out
            with nvtx.annotate(message=f"FFN (W2)", color="red"):
                w2_out = self.w2(silu_times_w3)
            return w2_out


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=4 * args.dim,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(
            self,
            x: torch.Tensor,
            start_pos: int,
            freqs_cis: torch.Tensor,
            mask: Optional[torch.Tensor],
    ):
        if config.get_use_triton():
            x_rms_norm = None
        else:
            with nvtx.annotate(message=f"RMS proj (element W)", color="blue"):
                x_rms_norm = self.attention_norm(x)

        h = x + self.attention.forward(
            x, x_rms_norm, start_pos, freqs_cis, mask, self.attention_norm.weight,
        )
        if config.get_use_triton():
            h_rms_norm = None
        else:
            with nvtx.annotate(message=f"RMS FF (element W)", color="blue"):
                h_rms_norm = self.ffn_norm(h)
        out = h + self.feed_forward.forward(h, h_rms_norm, self.ffn_norm.weight)
        return out


class Transformer(nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        self.tok_embeddings = torch.nn.Embedding(  # ParallelEmbedding(
            params.vocab_size, params.dim,  # init_method=lambda x: x
        )

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = torch.nn.Linear(  # ColumnParallelLinear(
            params.dim, params.vocab_size, bias=False,  # init_method=lambda x: x
        )

        self.freqs_cis = precompute_freqs_cis(
            self.params.dim // self.params.n_heads, self.params.max_seq_len * 2
        )
        # to reduce overhead we disable nvtx context manager when not required
        if not config.get_nvtx():
            nvtx.annotate = NoOpContextManager

    @torch.inference_mode()
    def forward(self, tokens: torch.Tensor, start_pos: int):
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[start_pos: start_pos + seqlen]

        mask = None
        if seqlen > 1:
            mask = torch.full(
                (1, 1, seqlen, seqlen), float("-inf"), device=tokens.device
            )
            mask = torch.triu(mask, diagonal=start_pos + 1).type_as(h)

        for layer in self.layers:
            with nvtx.annotate(message=f"layer {layer.layer_id} (element W + matmul)", color="white"):
                h = layer(h, start_pos, freqs_cis, mask)
        h = self.norm(h)
        with nvtx.annotate(message=f"output proj", color="red"):
            output = self.output(h)
        output = output.float()
        return output
