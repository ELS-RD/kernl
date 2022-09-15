import os
from pathlib import Path
from typing import List

import torch
from onnxruntime.transformers.optimizer import optimize_model
from transformers import AutoModel
from transformers import AutoTokenizer
from transformers.onnx import export, FeaturesManager

from test.models.ort_utils import create_model_for_provider


def build_onnx(model_name: str, model_path: str) -> [List[str], List[str]]:
    onnx_path_model = os.path.join(model_path, f"{model_name}.onnx")
    if os.path.exists(onnx_path_model):
        onnx_model = create_model_for_provider(onnx_path_model)
        return onnx_model
    model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float32)
    model.to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    onnx_path = Path(onnx_path_model)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    model_kind, model_onnx_config = FeaturesManager.check_supported_model_or_raise(model)
    onnx_config = model_onnx_config(model.config)
    _ = export(preprocessor=tokenizer, model=model, config=onnx_config, opset=onnx_config.default_onnx_opset, output=onnx_path, device="cuda")
    onnx_model = create_model_for_provider(onnx_path.as_posix())
    return onnx_model


def optimize_onnx(model_name: str, model_path: str, float16: bool = False, model_type: str = "bert", hidden_size: int = 0, num_heads: int = 0):
    optim_model_name = f"{model_name}_optim_fp16.onnx" if float16 else f"{model_name}_optim_fp32.onnx"
    if os.path.exists(os.path.join(model_path, optim_model_name)):
        optimized_model = create_model_for_provider(os.path.join(model_path, optim_model_name))
        return optimized_model
    if not os.path.exists(os.path.join(model_path, model_name)):
        _ = build_onnx(model_name, model_path)
    opt_level = 1 if model_type == "bert" else 0
    optimized_model = optimize_model(
        os.path.join(model_path, f"{model_name}.onnx"),
        model_type,
        num_heads=num_heads,
        hidden_size=hidden_size,
        opt_level=opt_level,
        use_gpu=True,
        only_onnxruntime=True,
    )

    if float16:
        optimized_model.convert_float_to_float16(use_symbolic_shape_infer=False)

    optimized_model.save_model_to_file(os.path.join(model_path, optim_model_name))
    return create_model_for_provider(os.path.join(model_path, optim_model_name))
