from typing import List

import torch
from transformers import AutoModel, PreTrainedModel
import torchdynamo
from torch.fx import replace_pattern
from torchdynamo.optimizations import BACKENDS

from implementations.attention import attention_forward


def attention_wrapper(q, k, v, sm_scale, is_causal, *args):
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    return attention_forward(q, k, v, sm_scale, is_causal=is_causal)


torch.fx.wrap('attention_wrapper')


def get_model_baseline():
    model_name = "bert-base-uncased"
    model = AutoModel.from_pretrained(pretrained_model_name_or_path=model_name, torch_dtype=torch.float16)
    return model.eval().cuda()


def get_model_dynamo():
    base = get_model_baseline()

    def compiler(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
        return gm.forward  # return a python callable

    def run(*args, **kwargs):
        with torchdynamo.optimize(compiler):
            return base(*args, **kwargs)

    return run


def remove_dropout(gm: torch.fx.GraphModule):
    for n in gm.graph.nodes:
        # If the target matches one of the patterns
        if "attention_self_dropout" in n.name:
            # Set the insert point, add the new node, and replace all uses
            # of `n` with the new node
            with gm.graph.inserting_after(n):
                # new_node = gm.graph.call_function(torch.nn.Identity, n.args, n.kwargs)
                n.replace_all_uses_with(n.args[0])
            # Remove the old node from the graph
            gm.graph.erase_node(n)
    gm.recompile()


def get_model_dynamo_nvfuser_ofi():
    base = get_model_baseline()

    def compiler(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
        compiled = BACKENDS["nvfuser_ofi"](gm, example_inputs)
        return compiled

    def run(*args, **kwargs):
        with torchdynamo.optimize(compiler):
            return base(*args, **kwargs)

    return run


def get_model_dynamo_cudagraphs():
    base = get_model_baseline()

    def compiler(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
        compiled = BACKENDS["cudagraphs"](gm, example_inputs)
        return compiled

    def run(*args, **kwargs):
        with torchdynamo.optimize(compiler):
            return base(*args, **kwargs)

    return run


def get_model_dynamo_droput_removed():
    base = get_model_baseline()

    def compiler(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
        remove_dropout(gm)

        return gm.forward  # return a python callable

    def run(*args, **kwargs):
        with torchdynamo.optimize(compiler):
            return base(*args, **kwargs)

    return run


def attention_fusion(gm: torch.fx.GraphModule, is_causal: bool):
    def pattern(permute_42, permute_40, attention_mask, permute_41):
        transpose_10 = permute_40.transpose(-1, -2)
        matmul_20 = torch.matmul(permute_42, transpose_10)
        truediv_10 = matmul_20 / 8.0
        add_30 = truediv_10 + attention_mask
        softmax_10 = torch.nn.functional.softmax(add_30, dim=-1)
        matmul_21 = torch.matmul(softmax_10, permute_41)
        return matmul_21

    def replace(permute_42, permute_40, attention_mask, permute_41):
        return attention_wrapper(permute_42, permute_40, permute_41, 1 / 8.0, is_causal, attention_mask)

    remove_dropout(gm)
    replace_pattern(gm, pattern, replace)
    gm.recompile()


def get_model_dynamo_fused_attention(is_causal=False):
    base = get_model_baseline()

    def compiler(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
        attention_fusion(gm, is_causal)
        return gm  # return a python callable

    def run(*args, **kwargs):
        with torchdynamo.optimize(compiler):
            return base(*args, **kwargs)

    return run


def get_model_dynamo_fused_attention_plus_dynamo_cudagraphs(is_causal=False):
    base = get_model_baseline()

    def compiler(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
        attention_fusion(gm, is_causal)
        compiled = BACKENDS["cudagraphs"](gm, example_inputs)
        return compiled  # return a python callable

    def run(*args, **kwargs):
        with torchdynamo.optimize(compiler):
            return base(*args, **kwargs)

    return run
