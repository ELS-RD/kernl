from typing import Callable, Union

import torch


def cuda_graphs_wrapper(model: Callable, inputs: Union[list[torch.Tensor], tuple[torch.Tensor]], copy_outputs: bool = False, pool: (int, int) = torch.cuda.graph_pool_handle()):
    """
    From torchdynamo
    """
    assert isinstance(inputs, (list, tuple)), f"inputs is of type {type(inputs)} instead of list"
    static_inputs = [torch.zeros_like(x) for x in inputs]
    # required warmup, not just for perf but for correctness
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        # 2 rounds, 1 to build the model (triton kernels, casting, etc.),
        # and 1 for warmup
        for _ in range(2):
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

    def run(*new_inputs):
        assert isinstance(new_inputs, (list, tuple)), f"inputs is of type {type(new_inputs)} instead of list"
        assert len(static_inputs) == len(new_inputs), f"{len(static_inputs)} == {len(new_inputs)}"
        for dst, src in zip(static_inputs, new_inputs):
            dst.copy_(src)  # cuda graph can only read data from the same address
        graph.replay()
        if copy_outputs:
            return [x.clone() for x in static_outputs]
        else:
            return static_outputs

    return run
