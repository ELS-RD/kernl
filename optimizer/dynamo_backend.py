import torch

from optimizer.attention import fuse_attention
from optimizer.dropout import remove_dropout
from optimizer.layer_norm import replace_layer_norm
from optimizer.linear import replace_all_linear


def dynamo_backend_ofi(gm: torch.fx.GraphModule, assume_causal=False):
    remove_dropout(gm)
    fuse_attention(gm, assume_causal)

    try:
        replace_all_linear(gm)
    except Exception as err:
        print(err)

    try:
        replace_layer_norm(gm)
    except Exception as err:
        print(err)

    return gm
