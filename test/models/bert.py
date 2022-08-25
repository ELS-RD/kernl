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
        return gm  # return a python callable

    def run(*args, **kwargs):
        with torchdynamo.optimize(compiler):
            return base(*args, **kwargs)

    return run


def get_model_optimized_without_cudagraph():
    base = get_model_baseline()

    def compiler(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
        dynamo_backend_ofi(gm)
        return gm  # return a python callable

    return torchdynamo.optimize(compiler)(base)


def cudagraphs_inner(model, inputs, copy_outputs=True):
    assert isinstance(inputs, (list, tuple))
    static_inputs = [torch.zeros_like(x) for x in inputs]

    # warmup
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        model(*inputs)
    stream.synchronize()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()

    # record
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        static_outputs = model(*static_inputs)
    if not isinstance(static_outputs, (list, tuple)):
        static_outputs = (static_outputs,)

    def run(*new_inputs):
        assert len(static_inputs) == len(new_inputs)
        for dst, src in zip(static_inputs, new_inputs):
            dst.copy_(src)
        graph.replay()
        if copy_outputs:
            return [x.clone() for x in static_outputs]
        else:
            return static_outputs

    return run


def get_model_optimized():
    base = get_model_baseline()

    def compiler(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
        dynamo_backend_ofi(gm)
        return cudagraphs_inner(gm, example_inputs)

    gm = torchdynamo.optimize(compiler)(base)
    return gm


def get_model_optimized_causal():
    base = get_model_baseline()

    def compiler(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
        dynamo_backend_ofi(gm, assume_causal=True)
        return cudagraphs_inner(gm, example_inputs)

    gm = torchdynamo.optimize(compiler)(base)
    return gm
