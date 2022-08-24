from typing import List

import torch
from transformers import AutoModel
import torchdynamo

from optimizer.dropout import remove_dropout
from optimizer.dynamo_backend import dynamo_backend_ofi
from torchdynamo.optimizations import BACKENDS


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


def get_model_dynamo_nvfuser_ofi():
    base = get_model_baseline()

    def compiler(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
        compiled = BACKENDS["nvfuser_ofi"](gm, example_inputs)
        return compiled

    def run(*args, **kwargs):
        with torchdynamo.optimize(compiler):
            return base(*args, **kwargs)

    return run
def get_model_dynamo_inductor():
    base = get_model_baseline()
    @torchdynamo.optimize("inductor")
    def run(*args, **kwargs):
        return base(*args, **kwargs)

    return run

def get_model_dynamo_onnx2tensorrt():
    base = get_model_baseline()
    @torchdynamo.optimize("onnx2tensorrt")
    def run(*args, **kwargs):
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


def get_model_dynamo_dropout_removed():
    base = get_model_baseline()

    def compiler(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
        remove_dropout(gm)
        return gm.forward  # return a python callable

    def run(*args, **kwargs):
        with torchdynamo.optimize(compiler):
            return base(*args, **kwargs)

    return run


def get_model_optimized_without_cudagraph():
    base = get_model_baseline()

    def compiler(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
        dynamo_backend_ofi(gm, example_inputs, enable_cudagraph=False)
        return gm.forward # return a python callable

    return torchdynamo.optimize(compiler)(base)


def get_model_optimized():
    base = get_model_baseline()

    def compiler(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
        dynamo_backend_ofi(gm, example_inputs)
        return gm.forward  # return a python callable

    return torchdynamo.optimize(compiler)(base)

def get_model_optimized_causal():
    base = get_model_baseline()

    def compiler(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
        dynamo_backend_ofi(gm, example_inputs, assume_causal=True)
        return gm.forward

    return torchdynamo.optimize(compiler)(base)
