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
import math
from typing import Callable, Union

import torch
from torch._dynamo import utils as dynamo_utils
from torch._inductor.compile_fx import cudagraphify_impl
from torch._subclasses import FakeTensor

from kernl.optimizer.pool_cuda_graphs import CudaGraphPool, get_aligned_size


# avec :
# time to warmup: 12.41min
# Kernl speedup: 2.5X (0.5 VS 1.3 min)
# # different outputs: 0/73 (0.00%)
#
# memory footprint:
# * allocated: 8.6GB
# * reserved: 11.3GB
# * max reserved: 13.6GB
#
# sans :
# time to warmup: 12.42min
# Kernl speedup: 2.4X (0.5 VS 1.3 min)
# # different outputs: 0/73 (0.00%)
#
# memory footprint:
# * allocated: 10.9GB
# * reserved: 13.4GB
# * max reserved: 13.9GB
#
# =======================================================================
# 2859 passed, 80 warnings in 8342.38s (2:19:02)
# ========================================================================
#
static_inputs_pool = []


def prepare_inputs(inputs: list[torch.Tensor], pools: list[CudaGraphPool]) -> list[torch.Tensor]:
    """
    Copy the inputs in the CUDA graphs memory pool and return tensor copies.
    Follows a greedy bin packing algorithm (first-fit decreasing) to minimize the number of pools:
    - sort the items in decreasing order of size ;
    - insert them one by one into the first bin that has room for it.

    :param inputs: list of tensors to copy in the pool
    :param pools: list of available pools
    :return: list of tensors that are copies of the inputs in the pool
    """
    inputs_copy = list(inputs).copy()
    # add position meta
    for index, t in enumerate(inputs_copy):
        t.position = index

    # reset pool offsets
    for p in pools:
        p.reset()

    pools.sort(key=lambda x: x.size, reverse=False)
    inputs_copy.sort(key=lambda x: x.numel(), reverse=True)

    to_add_new_pool: list[torch.Tensor] = list()
    outputs: list[torch.Tensor] = list()
    for t in inputs_copy:
        new_pool = True
        for pool in pools:
            if pool.can_store(t):
                new_pool = False
                new_t = pool.copy_to_pool(t)
                new_t.position = t.position
                outputs.append(new_t)
                break
        if new_pool:
            to_add_new_pool.append(t)

    if len(to_add_new_pool) > 0:
        total_input_size = sum([get_aligned_size(t) for t in to_add_new_pool])
        if len(static_inputs_pool) > 0:
            total_input_size = max(total_input_size, *([p.size for p in static_inputs_pool]))
        if total_input_size < 1024 * 1024 * 1024:
            total_input_size = 2 ** math.ceil(math.log2(total_input_size))
        new_pool = CudaGraphPool(total_input_size, device=inputs[0].device)  # size in bytes
        pools.append(new_pool)

        for t in to_add_new_pool:
            assert new_pool.can_store(t)
            new_t = new_pool.copy_to_pool(t)
            new_t.position = t.position
            outputs.append(new_t)

    outputs.sort(key=lambda x: x.position, reverse=False)
    return outputs


def cuda_graphs_wrapper(model: Callable, inputs: Union[list[torch.Tensor], tuple[torch.Tensor]]) -> Callable:
    """
    Wrapper to run the model with cuda graphs.
    @param model: model to save as a CUDA graph
    @param inputs: inputs to the model
    @return: an inference function that runs the model with cuda graphs
    """

    assert isinstance(inputs, (list, tuple))
    # if using fake tensors, defer CUDA graphs until we get real inputs at runtime
    if not any(isinstance(inp, FakeTensor) for inp in inputs):
        inputs = prepare_inputs(inputs=inputs, pools=static_inputs_pool)
        model(*inputs)  # additional warmup needed when input is mutated by some kernel
        f = cudagraphify_impl(
            model=lambda args: model(*args), inputs=inputs, static_input_idxs=tuple(range(len(inputs)))
        )
        return lambda *args: f(prepare_inputs(inputs=args, pools=static_inputs_pool))

    compiled_fn = None

    def run(*new_inputs):
        new_inputs = prepare_inputs(inputs=list(new_inputs), pools=static_inputs_pool)
        nonlocal compiled_fn
        if compiled_fn is None:
            with dynamo_utils.preserve_rng_state():
                model(*new_inputs)  # additional warmup needed when input is mutated by some kernel
                f = cudagraphify_impl(
                    model=lambda args: model(*args), inputs=new_inputs, static_input_idxs=tuple(range(len(inputs)))
                )

                def compiled_fn(args):
                    return f(list(args))

        return compiled_fn(new_inputs)

    return run
