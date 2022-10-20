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
from typing import Callable, List

import torch
import torchdynamo
from transformers import PreTrainedModel

from kernl.implementations.cuda_graph import cuda_graphs_wrapper
from kernl.optimizer.dynamo_backend import dynamo_backend_ofi


def optimize_model(original_model: PreTrainedModel, handle_graph_pool: bool = True) -> Callable:
    """
    Optimizes a given model. Optimization is done in two steps:
    *  first step is to convert the given model to fx graph.
    *  second step is to replace patterns found in the graph in order to optimize the model.

    @return: returns the optimized model (and the original model is modified, so it can not be used after optimization).

        Example:
            from kernl.model_optimization import optimize_model

            model_name = "BaptisteDoyen/camembert-base-xnli"
            model = AutoModelForSequenceClassification.from_pretrained(model_name)
            model = model.eval().cuda()

            optimized_model = optimize_model(model)
    """
    if handle_graph_pool:
        pool: (int, int) = torch.cuda.graph_pool_handle()
        pool_is_defined = True
    original_model.forward2 = original_model.forward

    def compiler(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
        dynamo_backend_ofi(gm)
        return (
            cuda_graphs_wrapper(gm, example_inputs, pool=pool)
            if pool_is_defined
            else cuda_graphs_wrapper(gm, example_inputs)
        )

    def run(*args, **kwargs):
        with torchdynamo.optimize(compiler):
            return original_model.forward2(*args, **kwargs)

    def forward_with_exception(*args, **kwargs):
        raise Exception("Original model can not be used after optimization")

    original_model.forward = forward_with_exception

    return run
