import tempfile
from typing import List

import torch
import torchdynamo
from torchdynamo.optimizations import BACKENDS
from transformers import AutoModel

#from implementations.cuda_graph import cuda_graphs_wrapper
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


def get_bert_tensorrt_shapes():
    from test.models.trt_utils import TensorRTShape
    input_id_shape = TensorRTShape(
        min_shape=[1, 16], optimal_shape=[16, 512], max_shape=[16, 512], input_name="input_ids"
    )
    attention_mask_shape = TensorRTShape(
        min_shape=[1, 16], optimal_shape=[16, 512], max_shape=[16, 512], input_name="attention_mask"
    )
    token_type_id_shape = TensorRTShape(
        min_shape=[1, 16], optimal_shape=[16, 512], max_shape=[16, 512], input_name="token_type_ids"
    )
    input_shapes = [input_id_shape, attention_mask_shape, token_type_id_shape]
    output_shape = TensorRTShape(
        min_shape=[1],
        optimal_shape=[1],
        max_shape=[1],
        input_name="last_hidden_state",
    )
    output_shapes = [output_shape]
    return input_shapes, output_shapes


def get_bert_tensorrt():
    from test.models.trt_utils import get_model_tensorrt
    input_shapes, output_shapes = get_bert_tensorrt_shapes()
    return get_model_tensorrt(model_name, models_dir, input_shapes, output_shapes, fp16_layer_selection=True)


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