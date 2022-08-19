import triton

from implementations.batched_matmul import matmul_kernel

from .kernel_wrapper import CudaGraphKernelWrapper, Tensors
from .static_tensor import CudaGraphStaticTensor, CudaGraphStaticTensors


class CudaGraphMatmulWrapper(CudaGraphKernelWrapper):
    def declare_tensors(self, *args, **kwargs) -> CudaGraphStaticTensors:
        a = args[0]
        tensors: CudaGraphStaticTensors = dict()
        static_a = CudaGraphStaticTensor("a", a.device, a.dtype)
        static_b = CudaGraphStaticTensor("b", a.device, a.dtype)
        static_c = CudaGraphStaticTensor("c", a.device, a.dtype)
        tensors = dict()
        tensors["a"] = static_a
        tensors["b"] = static_b
        tensors["c"] = static_c
        return tensors

    def prepare_tensors(self, tensors: CudaGraphStaticTensors, input: Tensors):
        """
        Generate a view of both static tensors for the graph.
        :param inputs: original input
        """
        static_a = tensors["a"]
        static_b = tensors["b"]
        static_c = tensors["c"]
        static_a.update_content(input["a"])
        static_b.update_content(input["b"])
        batch_size, M, _ = static_a.size
        _, _, N = static_b.size
        static_c.update_size((batch_size, M, N))

    def run_kernel(self, tensors: CudaGraphStaticTensors, *args, **kwargs):
        # checks constraints
        assert "a" in tensors
        assert "b" in tensors
        assert "c" in tensors
        a = tensors["a"]
        b = tensors["b"]
        c = tensors["c"]
        assert a.size[2] == b.size[1], "incompatible dimensions"
        assert a.tensor.is_contiguous(), "matrix A must be contiguous"
        assert b.tensor.is_contiguous(), "matrix B must be contiguous"
        batch_size, M, K = a.size
        _, K, N = b.size
        # assert (
        #         K % 32 == 0
        # ), "We don't check memory-out-of-bounds with K so K must be divisible by BLOCK_SIZE_K"
        # allocates output

        # 1D launch kernel where each block gets its own program.
        grid = lambda META: (
            triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
            batch_size,
        )
        matmul_kernel[grid](
            a.tensor,
            b.tensor,
            c.tensor,
            M,
            N,
            K,
            a.tensor.stride(0),
            a.tensor.stride(1),
            a.tensor.stride(2),
            b.tensor.stride(0),
            b.tensor.stride(1),
            b.tensor.stride(2),
            c.tensor.stride(0),
            c.tensor.stride(1),
            c.tensor.stride(2),
        )
        return c

    def prepare_output(self, tensors: CudaGraphStaticTensors) -> Tensors:
        return {"c": tensors["c"].tensor}
