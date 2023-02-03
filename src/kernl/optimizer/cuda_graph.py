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
from typing import Callable, Iterable, Optional, Union

import torch
from torch._dynamo import utils as dynamo_utils
from torch._inductor.compile_fx import cudagraphify_impl
from torch._subclasses import FakeTensor

from kernl.optimizer.pool_cuda_graphs import CudaGraphPool, get_aligned_size


static_inputs_pool = []


def get_pool_size(inputs: list[torch.Tensor], existing_pools: list[CudaGraphPool]) -> int:
    """
    Get the size of the pool to use for the CUDA graphs:
    - pool size should be at least as big as the largest existing pool size
    - if pool size < 1Gb, increase its size up to next power of 2 to avoid having many unusuable small pools

    :param inputs: list of inputs to be copied in the pool
    :param existing_pools: list of existing pools
    :return: size of the pool in bytes
    """
    size = sum([get_aligned_size(p) for p in inputs])
    size = max(size, *([p.size for p in existing_pools] + [0]))

    if size < 1024 * 1024 * 1024:
        size = 2 ** math.ceil(math.log2(size))
    return size


def argsort(iterable: Iterable, key: Callable) -> list[int]:
    """
    Sort the list of tensors following provided lambda function.
    :param iterable: iterable object to sort
    :param key: lambda function to sort the iterable object
    :return: indices to sort the iterable object
    """
    return [idx for idx, _ in sorted(enumerate(iterable), key=lambda x: key(x[1]))]


def prepare_inputs(inputs: list[torch.Tensor], pools: list[CudaGraphPool]) -> list[torch.Tensor]:
    """
    Copy the inputs in the CUDA graphs memory pool and return tensor copies.
    Follows a greedy bin packing algorithm (first-fit decreasing) to minimize the number of pools:
    - sort the items in decreasing order of size ;
    - insert them one by one into the first bin that has room for it.

    :param inputs: list of tensors to copy in the pool
    :param pools: list of available pools
    :return: copy of input tensors having their underlying storage in the memory pool
    """
    # reset pool offsets
    for p in pools:
        p.reset()

    pools.sort(key=lambda x: x.size, reverse=False)
    inputs_size_order = argsort(inputs, key=lambda x: x.untyped_storage().size())

    input_copies: list[Optional[torch.Tensor]] = [None] * len(inputs)
    new_pool_index = list()
    for idx in inputs_size_order:
        t = inputs[idx]
        new_pool = True
        for pool in pools:
            if pool.can_store(t):
                new_pool = False
                new_t = pool.copy_to_pool(t)
                input_copies[idx] = new_t
                break
        if new_pool:
            new_pool_index.append(idx)

    if len(new_pool_index) > 0:
        pool_size = get_pool_size(inputs=[inputs[i] for i in new_pool_index], existing_pools=pools)
        new_pool = CudaGraphPool(pool_size, device=inputs[0].device)
        pools.append(new_pool)

        for idx in new_pool_index:
            t = inputs[idx]
            assert new_pool.can_store(t)
            new_t = new_pool.copy_to_pool(t)
            input_copies[idx] = new_t

    return input_copies


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
