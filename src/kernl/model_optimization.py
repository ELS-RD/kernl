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
from typing import List

import torch
import torch._dynamo as torchdynamo
from transformers import PreTrainedModel

from kernl.optimizer.cuda_graph import cuda_graphs_wrapper, static_inputs_pool
from kernl.optimizer.dynamo_backend import dynamo_backend_ofi


# needs to be generated once to be reused several times, like encoder/decoder models
# https://github.com/pytorch/torchdynamo/issues/1816
def _compiler(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
    dynamo_backend_ofi(gm)
    return cuda_graphs_wrapper(gm, example_inputs)


def optimize_model(model: PreTrainedModel) -> None:
    """Optimizes a given model by replacing forward method by a call to optimized code.
    It is done in two steps:

    *  first step is to convert the given model to fx graph.
    *  second step is to replace patterns found in the graph by fast to run kernels.

    Examples:

        ``` { .py }
        import tensorflow as tf

        model = AutoModel.from_pretrained(...).eval().cuda()

        optimize_model(model)
        inputs = ...
        model(**inputs)
        ```

    Args:
        model: model to optimize
    """
    assert torch.cuda.is_available(), "CUDA capacity is required to use Kernl"
    major, _ = torch.cuda.get_device_capability()
    if major < 8:
        raise RuntimeError("GPU compute capability 8.0 (Ampere) or higher is required to use Kernl")
    assert next(model.parameters()).device.type == "cuda", "Model must be on GPU"
    static_inputs_pool.clear()
    model.forward_original = model.forward

    @torchdynamo.optimize(_compiler)
    def run(*args, **kwargs):
        return model.forward_original(*args, **kwargs)

    model.forward = run
