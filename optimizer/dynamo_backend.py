import torch

from optimizer.attention import fuse_attention, fuse_attention_2
from optimizer.dropout import remove_dropout
from optimizer.layer_norm import replace_layer_norm
from optimizer.linear import replace_all_linear
from optimizer.normalizer import normalize_operators


def dynamo_backend_ofi(gm: torch.fx.GraphModule, assume_causal=False):
    normalize_operators(gm)
    remove_dropout(gm)
    fuse_attention(gm, assume_causal)
    fuse_attention_2(gm, assume_causal)
    replace_all_linear(gm)
    replace_layer_norm(gm)
    return gm
