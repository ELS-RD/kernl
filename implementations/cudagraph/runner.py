import logging
from typing import Dict, Tuple

import torch

from .kernel_wrapper import CudaGraphKernelWrapper, Tensors
from .static_tensor import CudaGraphStaticTensors


CudaGraphs = Dict[Tuple, torch.cuda.graphs.CUDAGraph]


class CudaGraphRunner:
    def __init__(self, kernel: CudaGraphKernelWrapper, static_tensors: CudaGraphStaticTensors, warmup: int = 10):
        self.kernel: CudaGraphKernelWrapper = kernel
        self.tensors: CudaGraphStaticTensors = static_tensors
        self.graphs: CudaGraphs = dict()
        self.mempool = torch.cuda.graph_pool_handle()
        self.warmup: int = warmup

    def _needs_recompilation(self):
        for tensor_name in self.tensors:
            if self.tensors[tensor_name].needs_graph_recompilation:
                return True
        return False

    def _compile(self) -> torch.cuda.graphs.CUDAGraph:
        # change CUDA stream
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(self.warmup):
                self.kernel.run_kernel(self.tensors)
        # set back default CUDA stream
        torch.cuda.current_stream().wait_stream(stream)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=self.mempool):
            self.kernel.run_kernel(tensors=self.tensors)
        return graph

    def run(self, input: Tensors) -> Tensors:

        input_shape = tuple((name, input[name].size()) for name in sorted(input.keys()))

        self.kernel.prepare_tensors(self.tensors, input)

        if self._needs_recompilation():
            for shape in self.graphs:
                self.graphs[shape] = self._compile()
            for tensor_name in self.tensors:
                self.tensors[tensor_name].needs_graph_recompilation = False

        if input_shape in self.graphs:
            graph = self.graphs[input_shape]
        else:
            logging.info(f"compiling graph for input {input_shape})")
            graph = self._compile()
            self.graphs[input_shape] = graph
        graph.replay()

        output = self.kernel.prepare_output(self.tensors)

        return output
