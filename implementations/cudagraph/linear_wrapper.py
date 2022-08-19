import triton

from implementations.linear_layer import kernel_fma

from .kernel_wrapper import CudaGraphKernelWrapper, Tensors
from .static_tensor import CudaGraphStaticTensor, CudaGraphStaticTensors


class CudaGraphLinearWrapper(CudaGraphKernelWrapper):
    def declare_tensors(self, *args, **kwargs) -> CudaGraphStaticTensors:
        input = args[0]
        weights = args[1]
        tensors: CudaGraphStaticTensors = dict()
        static_input = CudaGraphStaticTensor("input", input.device, input.dtype)
        static_weights = CudaGraphStaticTensor("weights", weights.device, weights.dtype)
        static_output = CudaGraphStaticTensor("output", input.device, input.dtype)
        static_weights.update_content(weights)
        tensors = dict()
        tensors["input"] = static_input
        tensors["weights"] = static_weights
        tensors["output"] = static_output
        return tensors

    def run_kernel(self, tensors: CudaGraphStaticTensors, *args, **kwargs):
        assert "input" in tensors
        assert "weights" in tensors
        assert "output" in tensors
        input = tensors["input"]
        weights = tensors["weights"]
        output = tensors["output"]
        M, _ = input.size
        N, K = weights.size

        # 1D launch kernel where each block gets its own program.
        grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),)  # noqa
        BLOCK_K = 32 if K < 1024 else 64
        GROUP_M = 8  # speed optimization: group the programs

        kernel_fma[grid](
            output.tensor,
            input.tensor,  # act_inputs not used
            input.tensor,
            weights.tensor,  # data ptrs
            input.tensor,  # auto skip bias if not present
            M,
            N,
            K,  # shapes
            output.tensor.stride(0),
            input.tensor.stride(0),  # strides
            weights.tensor.stride(0),
            ACTIVATION=None,  # optional fused activation
            BIAS=False,  # optional fused bias
            GROUP_M=GROUP_M,
            BLOCK_K=BLOCK_K,
            SAVE_ACT_INPUTS=False,
        )

    def prepare_tensors(self, tensors: CudaGraphStaticTensors, input: Tensors):
        """
        Generate a view of both static tensors for the graph.
        :param inputs: original input
        """
        static_input = tensors["input"]
        static_weights = tensors["weights"]
        static_output = tensors["output"]
        self.batch_size, self.M, _ = input["input"].size()
        input_ = input["input"].flatten(0, 1)
        static_input.update_content(input_)
        M, _ = static_input.size
        N, _ = static_weights.size
        static_output.update_size((M, N))

    def prepare_output(self, tensors: CudaGraphStaticTensors) -> Tensors:
        static_output = tensors["output"]
        static_weights = tensors["weights"]
        static_output.update_size((self.batch_size, self.M, static_weights.size[0]))
        return {"output": static_output.tensor}
