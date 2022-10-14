#  Copyright 2022 Lefebvre Sarrut
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#


from typing import Dict

import numpy as np
import torch

# noinspection PyUnresolvedReferences
from onnxruntime import GraphOptimizationLevel, InferenceSession, IOBinding, SessionOptions


def create_model_for_provider(path: str) -> InferenceSession:
    options = SessionOptions()
    options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    options.log_severity_level = 2
    return InferenceSession(path, options, providers=["CUDAExecutionProvider"])


def inference_onnx_binding(
    model_onnx: InferenceSession, binding: IOBinding, inputs: Dict[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:

    device = "cuda"
    device_id = 0
    bs, seq_len = inputs["input_ids"].shape
    for input_onnx in model_onnx.get_inputs():
        if input_onnx.name not in inputs:  # some inputs may be optional
            continue
        tensor: torch.Tensor = inputs[input_onnx.name]

        binding.bind_input(
            name=input_onnx.name,
            device_type=device,
            device_id=device_id,
            element_type=np.int64,
            shape=tuple(tensor.shape),
            buffer_ptr=tensor.data_ptr(),
        )
        inputs[input_onnx.name] = tensor

    outputs: dict[str, torch.Tensor] = dict()
    for out in model_onnx.get_outputs():

        if out.name == "last_hidden_state":
            output_tensor = torch.empty((bs, seq_len, 768), dtype=torch.float16, device="cuda")
        elif out.name == "1607":  # pooler_output
            output_tensor = torch.empty((bs, 768), dtype=torch.float16, device="cuda")
        else:
            raise ValueError(f"Unknown output name {out.name}")
        outputs[out.name] = output_tensor
        binding.bind_output(
            name=out.name,
            device_type=device,
            device_id=device_id,
            element_type=np.float16,
            shape=tuple(output_tensor.shape),
            buffer_ptr=output_tensor.data_ptr(),
        )

    model_onnx.run_with_iobinding(binding)

    return outputs