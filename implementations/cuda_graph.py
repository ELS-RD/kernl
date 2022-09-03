import logging
from typing import Callable

import triton
import torch

from implementations.linear_layer import kernel_fma


def cudagraphs_inner(model: Callable, inputs: list, copy_outputs=True, pool: (int, int) = torch.cuda.graph_pool_handle()):
    """
    From torchdynamo
    """
    assert isinstance(inputs, (list, tuple))
    static_inputs = [torch.zeros_like(x) for x in inputs]

    # warmup
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        model(*inputs)
    stream.synchronize()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()

    # record
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream, pool=pool):
        static_outputs = model(*static_inputs)
    if not isinstance(static_outputs, (list, tuple)):
        static_outputs = (static_outputs,)

    def run(new_inputs):
        assert len(static_inputs) == len(new_inputs)
        for dst, src in zip(static_inputs, new_inputs):
            dst.copy_(src)
        graph.replay()
        if copy_outputs:
            return [x.clone() for x in static_outputs]
        else:
            return static_outputs

    return run
