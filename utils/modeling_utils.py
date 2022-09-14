import logging
from typing import Dict

import torch

logger = logging.getLogger(__name__)


def get_input_causal(shape: (int, int)) -> Dict[str, torch.Tensor]:
    batch, seq_length = shape
    mask = torch.tril(torch.ones((batch, seq_length), dtype=torch.int64, device="cuda"))
    return {
        "input_ids": torch.randint(2, 1000, size=shape, dtype=torch.int64, device="cuda"),
        "attention_mask": mask,
        "token_type_ids": torch.ones(size=shape, dtype=torch.int64, device="cuda")
    }


def get_input_non_causal(shape: (int, int)) -> Dict[str, torch.Tensor]:
    return {
        "input_ids": torch.randint(2, 1000, size=shape, dtype=torch.int32, device="cuda"),
        "attention_mask": None,  # TODO None is not a correct value, no key at all would be better
    }


def get_model_onnx(model_name: str, model_path: str):
    try:
        from utils.onnx_utils import build_onnx
        from utils.ort_utils import inference_onnx_binding
        from transformers.modeling_outputs import BaseModelOutput
    except ImportError as e:
        error = f"It seems that you are missing some dependencies. \n {e}"
        raise ImportError(error)
    model_onnx = build_onnx(model_name, model_path)

    def run(*args, **kwargs):
        inputs = {
            "input_ids": kwargs["input_ids"].to("cuda"),
            "attention_mask": kwargs["attention_mask"].to("cuda"),
            "token_type_ids": kwargs["token_type_ids"].to("cuda")
        }
        outputs = inference_onnx_binding(model_onnx=model_onnx, inputs=inputs, device="cuda")
        return BaseModelOutput(last_hidden_state=outputs["last_hidden_state"].type(torch.float32))
    return run


def get_model_optim_fp32_onnx(model_name: str, model_path: str):
    try:
        from utils.onnx_utils import optimize_onnx
        from utils.ort_utils import inference_onnx_binding
        from transformers.modeling_outputs import BaseModelOutput
    except ImportError as e:
        error = f"It seems that you are missing some dependencies. \n {e}"
        raise ImportError(error)
    model_onnx = optimize_onnx(model_name, model_path)

    def run(*args, **kwargs):
        inputs = {
            "input_ids": kwargs["input_ids"].to("cuda"),
            "attention_mask": kwargs["attention_mask"].to("cuda"),
            "token_type_ids": kwargs["token_type_ids"].to("cuda")
        }
        outputs = inference_onnx_binding(model_onnx=model_onnx, inputs=inputs, device="cuda")
        return BaseModelOutput(last_hidden_state=outputs["last_hidden_state"].type(torch.float32))
    return run


def get_model_optim_fp16_onnx(model_name: str, model_path: str):
    try:
        from utils.onnx_utils import optimize_onnx
        from utils.ort_utils import inference_onnx_binding
        from transformers.modeling_outputs import BaseModelOutput
    except ImportError as e:
        error = f"It seems that you are missing some dependencies. \n {e}"
        raise ImportError(error)
    model_onnx = optimize_onnx(model_name, model_path, True)

    def run(*args, **kwargs):
        inputs = {
            "input_ids": kwargs["input_ids"].to("cuda"),
            "attention_mask": kwargs["attention_mask"].to("cuda"),
            "token_type_ids": kwargs["token_type_ids"].to("cuda")
        }
        outputs = inference_onnx_binding(model_onnx=model_onnx, inputs=inputs, device="cuda")
        return BaseModelOutput(last_hidden_state=outputs["last_hidden_state"].type(torch.float32))
    return run
