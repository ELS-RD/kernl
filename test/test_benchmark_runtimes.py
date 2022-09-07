from typing import Tuple, Dict

import torch
from transformer_deploy.backends.ort_utils import create_model_for_provider, optimize_onnx, inference_onnx_binding
from transformer_deploy.backends.pytorch_utils import convert_to_onnx, get_model_size
from transformer_deploy.backends.trt_utils import build_engine, save_engine, load_engine
from transformers import AutoModel, AutoConfig, PretrainedConfig

import tensorrt as trt
from tensorrt import ICudaEngine, Logger, Runtime


def get_pytorch_input(size: Tuple[int, int]) -> Dict[str, torch.Tensor]:
    batch, seq_length = size
    mask = torch.tril(torch.ones((batch, seq_length, seq_length), device="cuda"))
    return {
        "input_ids": torch.randint(2, 1000, size=size, dtype=torch.int32, device="cuda"),
        "attention_mask": mask,
    }


def get_model_baseline():
    model_name = "bert-base-uncased"
    model = AutoModel.from_pretrained(pretrained_model_name_or_path=model_name).eval().cuda()
    return model


def get_model_onnx():
    model_pytorch = get_model_baseline()
    onnx_model_path = "bert.onnx"
    model_inputs = get_pytorch_input((16, 256))
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


def test_results():
    model_baseline = get_model_baseline()
    model_onnx = get_model_onnx()
    model_trt = get_model_tensorrt()
    for shape in [(16, 128), (16, 256)]:
        pytorch_input = get_pytorch_input(shape)
        expected = model_baseline(**pytorch_input)
        result_onnx = inference_onnx_binding(
            model_onnx=model_onnx,
            inputs=pytorch_input,
            device="cuda",
        )
        assert torch.allclose(result_onnx["last_hidden_state"], expected["last_hidden_state"], atol=1e-1)
        result_trt = model_trt(pytorch_input)
        assert torch.allclose(result_trt["last_hidden_state"], expected["last_hidden_state"], atol=1e-1)
