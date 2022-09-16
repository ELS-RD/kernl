import tempfile
from typing import List

import torch
import torchdynamo
from torchdynamo.optimizations import BACKENDS
from transformers import AutoModel

from implementations.cuda_graph import cuda_graphs_wrapper
from optimizer.dropout import remove_dropout
from optimizer.dynamo_backend import dynamo_backend_ofi

model_name = "bert-base-uncased"
models_dir = tempfile.TemporaryDirectory().name


def get_model_baseline(float_16: bool = True):
    model_dtype = torch.float16 if float_16 else torch.float32
    model = AutoModel.from_pretrained(pretrained_model_name_or_path=model_name, torch_dtype=model_dtype)
    return model.eval().cuda()


def get_bert_onnx():
    from test.models.onnx_utils import get_model_onnx
    return get_model_onnx(model_name, models_dir)


def get_bert_optim_fp32_onnx():
    from test.models.onnx_utils import get_model_optim_fp32_onnx
    return get_model_optim_fp32_onnx(model_name, models_dir)


def get_bert_optim_fp16_onnx():
    from test.models.onnx_utils import get_model_optim_fp16_onnx
    return get_model_optim_fp16_onnx(model_name, models_dir)


def get_model_dynamo_dropout_removed():
    base = get_model_baseline()

    def compiler(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
        remove_dropout(gm)
        return gm  # return a python callable

    def run(*args, **kwargs):
        with torchdynamo.optimize(compiler):
            return base(*args, **kwargs)

    return run


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


def get_model_dynamo_cuda_graphs():
    base = get_model_baseline()

    def compiler(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
        compiled = BACKENDS["cudagraphs"](gm, example_inputs)
        return compiled

    def run(*args, **kwargs):
        with torchdynamo.optimize(compiler):
            return base(*args, **kwargs)

    return run


def get_model_optimized():
    base = get_model_baseline()

    def compiler(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
        dynamo_backend_ofi(gm)
        return gm  # return a python callable

    def run(*args, **kwargs):
        with torchdynamo.optimize(compiler):
            return base(*args, **kwargs)

    return run


def get_model_optimized_cuda_graphs():
    base = get_model_baseline()

    def compiler(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
        dynamo_backend_ofi(gm)
        return cuda_graphs_wrapper(gm, example_inputs)

    def run(*args, **kwargs):
        with torchdynamo.optimize(compiler):
            return base(*args, **kwargs)

    return run


def get_model_optimized_causal_cuda_graphs():
    base = get_model_baseline()

    def compiler(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
        dynamo_backend_ofi(gm, assume_causal=True)
        return cuda_graphs_wrapper(gm, example_inputs)

    def run(*args, **kwargs):
        with torchdynamo.optimize(compiler):
            return base(*args, **kwargs)

    return run