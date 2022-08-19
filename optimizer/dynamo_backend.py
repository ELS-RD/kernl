import torch

from optimizer.attention import fuse_attention
from optimizer.dropout import remove_dropout
from optimizer.linear import replace_all_linear
from torchdynamo.optimizations import BACKENDS


def dynamo_backend_ofi(gm: torch.fx.GraphModule, example_inputs, enable_cudagraph = True, assume_causal = False):
    remove_dropout(gm)
    fuse_attention(gm, assume_causal)
    try:
        replace_all_linear(gm)
    except Exception as err:
        print(err)

    if enable_cudagraph:
        gm = BACKENDS["cudagraphs"](gm, example_inputs)
    return gm