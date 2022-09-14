import logging
import os
from typing import Dict, Tuple

from transformers import AutoModel
import torch

logger = logging.getLogger(__name__)


def get_input_causal(shape: (int, int)) -> dict[str, torch.Tensor]:
    batch, seq_length = shape
    mask = torch.tril(torch.ones((batch, seq_length, seq_length), dtype=torch.int32, device="cuda"))
    return {
        "input_ids": torch.randint(2, 1000, size=shape, dtype=torch.int32, device="cuda"),
        "attention_mask": mask,
        "token_type_ids": torch.ones(size=shape, dtype=torch.int32, device="cuda")
    }


def get_input_non_causal(shape: (int, int)) -> dict[str, torch.Tensor]:
    return {
        "input_ids": torch.randint(2, 1000, size=shape, dtype=torch.int32, device="cuda"),
        "attention_mask": None,  # TODO None is not a correct value, no key at all would be better
    }


def get_model_onnx(model_name: str, model_dir_path: str, shape: Tuple[int, int]):
    try:
        from test.utils.pytorch_utils import convert_to_onnx
        from test.utils.ort_utils import create_model_for_provider
        from test.utils.ort_utils import inference_onnx_binding
        from transformers.modeling_outputs import BaseModelOutput
    except ImportError:
        logger.error(
            "It seems that onnx runtime is not yet installed. Onnx models will not be included in the benchmark."
        )
        return
    baseline_model = AutoModel.from_pretrained(pretrained_model_name_or_path=model_name)
    baseline_model = baseline_model.to("cuda")
    model_onnx_name = f"{model_name}.onnx"
    model_onnx_path = os.path.join(model_dir_path, model_onnx_name)
    model_inputs = get_input_causal(shape)
    convert_to_onnx(
        model_pytorch=baseline_model,
        output_path=model_onnx_path,
        inputs_pytorch=model_inputs,
        quantization=False,
        var_output_seq=True,
        output_names=["last_hidden_state"]
    )
    model_onnx = create_model_for_provider(model_onnx_path, "CUDAExecutionProvider")

    def run(*args, **kwargs):
        inputs = {
            "input_ids": kwargs["input_ids"].to("cuda"),
            "attention_mask": kwargs["attention_mask"].to("cuda"),
            "token_type_ids": kwargs["token_type_ids"].to("cuda")
        }
        outputs = inference_onnx_binding(model_onnx=model_onnx, inputs=inputs, device="cuda")
        return BaseModelOutput(last_hidden_state=outputs["last_hidden_state"].type(torch.float32))
    return run


def get_model_tensorrt(model_name: str, model_dir_path: str, shape: Tuple[int, int]):
    try:
        from tensorrt.tensorrt import Logger
        import tensorrt as trt
        from tensorrt import Runtime
        from tensorrt.tensorrt import ICudaEngine
        from test.utils.trt_utils import build_engine, save_engine, load_engine, TensorRTShape
    except ImportError:
        logger.error(
            "It seems that TensorRT is not yet installed. It is required to include TensorRT in benchmark."
            "Please find installation instruction on: "
            "https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html"
        )
        return
    if not os.path.exists(os.path.join(model_dir_path, f"{model_name}.onnx")):
        _ = get_model_onnx(model_name, model_dir_path, shape)
    trt_logger: Logger = trt.Logger(trt.Logger.ERROR)
    runtime: Runtime = trt.Runtime(trt_logger)
    trt_model_name = f"{model_name}.plan"
    model_onnx_name = f"{model_name}.onnx"
    input_id_shape = TensorRTShape(
        min_shape=[1, 16], optimal_shape=[shape[0], shape[1]], max_shape=[shape[0], shape[1]], input_name="input_ids"
    )
    attention_mask_shape = TensorRTShape(
        min_shape=[1, 16, 16], optimal_shape=[shape[0], shape[1], shape[1]], max_shape=[shape[0], shape[1], shape[1]], input_name="attention_mask"
    )
    token_type_id_shape = TensorRTShape(
        min_shape=[1, 16], optimal_shape=[shape[0], shape[1]], max_shape=[shape[0], shape[1]], input_name="token_type_ids"
    )
    input_shapes = [input_id_shape, attention_mask_shape, token_type_id_shape]
    output_shape = TensorRTShape(
        min_shape=[1],
        optimal_shape=[1],
        max_shape=[1],
        input_name="last_hidden_state",
    )
    shape_tensors = [output_shape]
    engine_path = os.path.join(model_dir_path, f"{trt_model_name}")
    engine: ICudaEngine = build_engine(
        runtime=runtime,
        onnx_file_path=os.path.join(model_dir_path, model_onnx_name),
        logger=trt_logger,
        workspace_size=20000 * 1024 ** 2,
        fp16=False,
        int8=False,
        input_shapes=input_shapes,
        shape_tensors=shape_tensors,
    )
    save_engine(engine, engine_path)
    model_trt = load_engine(runtime=runtime, engine_file_path=engine_path)

    def run(*args, **kwargs):
        inputs = {
            "input_ids": kwargs["input_ids"].to("cuda"),
            "attention_mask": kwargs["attention_mask"].to("cuda"),
            "token_type_ids": kwargs["token_type_ids"].to("cuda")
        }
        return model_trt(inputs=inputs)

    return run
