import torch

from nucle.implementations.attention import attention_forward
from nucle.utils.extended_matcher import replace_pattern


def attention_wrapper(q, k, v, output, sm_scale, is_causal, attention_mask):
    return attention_forward(q, k, v, output, sm_scale, is_causal=is_causal, attention_mask=attention_mask)


torch.fx.wrap("attention_wrapper")


def fuse_attention(gm: torch.fx.GraphModule, is_causal: bool):
    def pattern(q, k, attention_mask, v):
        transpose_10 = k.transpose(-1, -2)
        matmul_20 = torch.matmul(q, transpose_10)
        truediv_10 = matmul_20 / 8.0
        add_30 = truediv_10 + attention_mask
        softmax_10 = torch.nn.functional.softmax(add_30, dim=-1)
        matmul_21 = torch.matmul(softmax_10, v)
        return matmul_21

    def replace(q, k, attention_mask, v):
        output = torch.empty_like(q)
        output = attention_wrapper(q, k, v, output, 1 / 8.0, is_causal, attention_mask)
        return output

    replace_pattern(gm, pattern, replace)
