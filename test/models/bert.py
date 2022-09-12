from typing import List, Dict

import torch
from transformer_deploy.backends.ort_utils import create_model_for_provider, optimize_onnx
from transformer_deploy.backends.pytorch_utils import convert_to_onnx, get_model_size
from transformers import AutoModel, AutoConfig, PretrainedConfig
import torchdynamo

from implementations.cuda_graph import cuda_graphs_wrapper
from optimizer.dropout import remove_dropout
from optimizer.dynamo_backend import dynamo_backend_ofi
from torchdynamo.optimizations import BACKENDS


def get_input_causal(shape: (int, int)) -> Dict[str, torch.Tensor]:
    batch, seq_length = shape
    mask = torch.tril(torch.ones((batch, seq_length, seq_length), device="cuda"))
    return {
        "input_ids": torch.randint(2, 1000, size=shape, dtype=torch.int32, device="cuda"),
        "attention_mask": mask,
    }


def get_input_non_causal(shape: (int, int)) -> Dict[str, torch.Tensor]:
    return {
        "input_ids": torch.randint(2, 1000, size=shape, dtype=torch.int32, device="cuda"),
        "attention_mask": None,  # TODO None is not a correct value, no key at all would be better
    }


def get_model_baseline(float_16: bool = True):
    model_dtype = torch.float16 if float_16 else torch.float32
    model_name = "bert-base-uncased"
    model = AutoModel.from_pretrained(pretrained_model_name_or_path=model_name, torch_dtype=model_dtype)
    return model.eval().cuda()


def get_model_onnx():
    model_pytorch = get_model_baseline()
    onnx_model_path = "bert.onnx"
    model_inputs = get_input_causal((16, 256))
    convert_to_onnx(
        model_pytorch=model_pytorch,
        output_path=onnx_model_path,
        inputs_pytorch=model_inputs,
        quantization=False,
        var_output_seq=True,
    )
    model_onnx = create_model_for_provider(onnx_model_path, "CUDAExecutionProvider")
    return model_onnx


def get_model_onnx_optimized():
    _ = get_model_onnx()
    onnx_model_path = "model.onnx"
    onnx_optim_model_path = "model_optim_fp16.onnx"
    num_attention_heads, hidden_size = get_model_size(path="bert-base-uncased")
    model_config: PretrainedConfig = AutoConfig.from_pretrained(
        pretrained_model_name_or_path="bert-base-uncased"
    )
    optimize_onnx(
        onnx_path=onnx_model_path,
        onnx_optim_model_path=onnx_optim_model_path,
        fp16=True,
        use_cuda=True,
        num_attention_heads=num_attention_heads,
        hidden_size=hidden_size,
        architecture=model_config.model_type,
    )
    model_onnx_optim = create_model_for_provider(path=onnx_optim_model_path, provider_to_use="CUDAExecutionProvider")
    return model_onnx_optim


def get_model_tensorrt():
    try:
        from tensorrt.tensorrt import Logger
        import tensorrt as trt
        from tensorrt import Runtime
        from tensorrt.tensorrt import ICudaEngine
        from transformer_deploy.backends.trt_utils import build_engine, save_engine, load_engine
    except ImportError:
        raise ImportError(
            "It seems that TensorRT is not yet installed. It is required when to include TensorRT in benchmark."
            "Please find installation instruction on: "
            "https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html"
        )
    _ = get_model_onnx()
    trt_logger: Logger = trt.Logger(trt.Logger.ERROR)
    runtime: Runtime = trt.Runtime(trt_logger)
    trt_model_name = "bert.plan"
    engine: ICudaEngine = build_engine(
        runtime=runtime,
        onnx_file_path="bert.onnx",
        logger=trt_logger,
        workspace_size=20000 * 1024 ** 2,
        fp16=False,  # for tests only
        int8=False,
        min_shape=(1, 256),
        optimal_shape=(16, 256),
        max_shape=(16, 256)
    )
    save_engine(engine, trt_model_name)
    model_trt = load_engine(runtime=runtime, engine_file_path=trt_model_name)
    return model_trt


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