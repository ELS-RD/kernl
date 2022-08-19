import logging
import triton
import torch

from implementations.linear_layer import kernel_fma

from typing import Dict


class CudaGraph:
    """
    Remove CPU overhead by using cuda graphs.
    Cuda graph will save the cuda instructions going through the cuda stream and replay them with a single instruction.
    It introduces some constrains, one being to read/write data from the memory address used during the graph building.
    We work around this by using a large tensor shared by all graphs plus a view of it for each shape we need.
    """
    def __init__(self, weights: torch.Tensor, container_size: int = 30000, warmup: int = 10):
        assert weights.is_contiguous()
        self.graphs: Dict[torch.Size, torch.cuda.graphs.CUDAGraph] = dict()
        self.mempool = torch.cuda.graph_pool_handle()
        self.weights: torch.Tensor = weights
        self.static_input: torch.Tensor = torch.empty((container_size,), device=weights.device, dtype=weights.dtype)
        self.static_output: torch.Tensor = torch.empty((container_size,), device=weights.device, dtype=weights.dtype)
        self.N, self.K = self.weights.size()  # problem size
        self.warmup = warmup

    @property
    def M(self):
        return self.static_input.shape[0]

    def run_kernel(self):
        # 1D launch kernel where each block gets its own program.
        grid = lambda META: (triton.cdiv(self.M, META["BLOCK_M"]) * triton.cdiv(self.N, META["BLOCK_N"]),)  # noqa
        BLOCK_K = 32 if self.K < 1024 else 64
        GROUP_M = 8  # speed optimization: group the programs

        kernel_fma[grid](
            self.static_output,
            self.static_input,  # act_inputs not used
            self.static_input,
            self.weights,  # data ptrs
            self.static_input,  # auto skip bias if not present
            self.M, self.N, self.K,  # shapes
            self.static_output.stride(0), self.static_input.stride(0),  # strides
            self.weights.stride(0),
            ACTIVATION=None,  # optional fused activation
            BIAS=False,  # optional fused bias
            GROUP_M=GROUP_M,
            BLOCK_K=BLOCK_K,
            SAVE_ACT_INPUTS=False
        )

    def prepare_static_tensors(self, inputs: torch.Tensor):
        """
        Generate a view of both static tensors for the graph.
        :param inputs: original input
        """
        inputs_ = inputs.flatten(0, 1)
        self.static_input.resize_as_(inputs_)
        self.static_input.copy_(inputs_)
        self.static_output.resize_((self.M, self.N))

    def _compile(self) -> torch.cuda.graphs.CUDAGraph:
        assert (self.static_input.shape[1] == self.weights.shape[1]), f"Incompatible dimensions in between inputs and weight, {self.static_input.shape} - {self.weights.shape}"

        # change CUDA stream
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(self.warmup):
                self.run_kernel()
        # set back default CUDA stream
        torch.cuda.current_stream().wait_stream(s)

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, pool=self.mempool):
            self.run_kernel()
        return g

    def rebuild_all_graphs(self):
        logging.info("rebuild all graphs")
        for shape in self.graphs.keys():
            self.graphs[shape] = self._compile()

    def run(self, inputs: torch.Tensor) -> torch.Tensor:
        # if we can't store data in existing static tensor, we need bigger ones + all graphs reading from them
        if inputs.nelement() > self.static_input.nelement():
            new_shape = (inputs.nelement() * 2,)
            self.static_input = torch.empty(new_shape, device=inputs.device, dtype=inputs.dtype, requires_grad=False)
            self.static_output = torch.empty(new_shape, device=inputs.device, dtype=inputs.dtype, requires_grad=False)
            self.rebuild_all_graphs()

        self.prepare_static_tensors(inputs=inputs)

        if inputs.size() in self.graphs:
            g = self.graphs[inputs.size()]
        else:
            logging.info(f"compiling graph for shape {inputs.size()}")
            g = self._compile()
            self.graphs[inputs.size()] = g
        g.replay()
        self.static_output.resize_(inputs.size()[0], inputs.size()[1], self.N)
        return self.static_output
