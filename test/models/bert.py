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
from transformers import AutoModel

from kernl.optimizer.cuda_graph import cuda_graphs_wrapper
from kernl.optimizer.dynamo_backend import dynamo_backend_ofi


def get_model_from_hf(model_name):
    model = AutoModel.from_pretrained(pretrained_model_name_or_path=model_name)
    return model.eval().cuda()


def get_model_baseline(base):
    return base


def get_model_dynamo_cuda_graphs(base):
    def compiler(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
        return cuda_graphs_wrapper(gm, example_inputs)

    @torchdynamo.optimize(compiler)
    def run(*args, **kwargs):
        return base(*args, **kwargs)

    return run


def get_model_optimized(base):
    def compiler(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
        dynamo_backend_ofi(gm)
        return gm  # return a python callable

    @torchdynamo.optimize(compiler)
    def run(*args, **kwargs):
        return base(*args, **kwargs)

    return run


def get_model_optimized_cuda_graphs(base):
    def compiler(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
        dynamo_backend_ofi(gm)
        return cuda_graphs_wrapper(gm, example_inputs)

    @torchdynamo.optimize(compiler)
    def run(*args, **kwargs):
        return base(*args, **kwargs)

    return run


def get_model_optimized_causal_cuda_graphs(base):
    def compiler(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
        dynamo_backend_ofi(gm, assume_causal=True)
        return cuda_graphs_wrapper(gm, example_inputs)

    @torchdynamo.optimize(compiler)
    def run(*args, **kwargs):
        return base(*args, **kwargs)

    return run
