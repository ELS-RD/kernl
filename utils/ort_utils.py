"""
All the tooling to ease ONNX Runtime usage.
"""
import ctypes as C
from ctypes.util import find_library
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
# noinspection PyUnresolvedReferences
from onnxruntime import ExecutionMode, GraphOptimizationLevel, InferenceSession, IOBinding, OrtValue, SessionOptions

libc = C.CDLL(find_library("c"))
libc.malloc.restype = C.c_void_p

# GPU inference only
try:
    # noinspection PyUnresolvedReferences
    import cupy as cp
except ImportError:
    pass


def create_model_for_provider(
    path: str,
    provider_to_use: Union[str, List],
    nb_threads: int = 0,
    optimization_level: GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
    enable_profiling: bool = False,
    log_severity: int = 2,
) -> InferenceSession:
    """
    Create an ONNX Runtime instance.
    :param path: path to ONNX file or serialized to string model
    :param provider_to_use: provider to use for inference
    :param nb_threads: intra_op_num_threads to use. You may want to try different parameters,
        more core does not always provide best performances. 0 let ORT choose the best value.
    :param optimization_level: expected level of ONNX Runtime optimization. For GPU and NLP, extended is the one
        providing kernel fusion of element wise operations. Enable all level is for CPU inference.
        see https://onnxruntime.ai/docs/performance/graph-optimizations.html#layout-optimizations
    :param enable_profiling: let Onnx Runtime log each kernel time.
    :param log_severity: Log severity level. 0:Verbose, 1:Info, 2:Warning. 3:Error, 4:Fatal.
    :return: ONNX Runtime inference session
    """
    options = SessionOptions()
    options.graph_optimization_level = optimization_level
    options.enable_profiling = enable_profiling
    options.log_severity_level = log_severity
    if isinstance(provider_to_use, str):
        provider_to_use = [provider_to_use]
    if provider_to_use == ["CPUExecutionProvider"]:
        options.execution_mode = ExecutionMode.ORT_SEQUENTIAL
        options.intra_op_num_threads = nb_threads
    return InferenceSession(path, options, providers=provider_to_use)


# https://github.com/pytorch/pytorch/blob/ac79c874cefee2f8bc1605eed9a924d80c0b3542/torch/testing/_internal/common_utils.py#L349
numpy_to_torch_dtype_dict: Dict[np.dtype, torch.dtype] = {
    np.bool_: torch.bool,
    np.uint8: torch.uint8,
    np.int8: torch.int8,
    np.int16: torch.int16,
    np.int32: torch.int32,
    np.int64: torch.int64,
    np.float16: torch.float16,
    np.float32: torch.float32,
    np.float64: torch.float64,
    np.complex64: torch.complex64,
    np.complex128: torch.complex128,
}

torch_to_numpy_dtype_dict: Dict[torch.dtype, np.dtype] = {v: k for k, v in numpy_to_torch_dtype_dict.items()}

# np.ctypeslib.as_ctypes_type not used as it imply to manage exception, etc. for unsupported types like float16
ort_conversion_table: Dict[str, Tuple[torch.dtype, Optional[np.dtype], Optional[int], int]] = {
    # bool not supported by DlPack! see https://github.com/dmlc/dlpack/issues/75
    "tensor(bool)": (torch.bool, None, C.c_bool, 1),
    "tensor(int8)": (torch.int8, np.int8, C.c_int8, 1),
    "tensor(int16)": (torch.int16, np.int16, C.c_int16, 2),
    "tensor(int32)": (torch.int32, np.int32, C.c_int32, 4),
    "tensor(int64)": (torch.int64, np.int64, C.c_int64, 8),
    "tensor(float16)": (torch.float16, np.float16, None, 2),
    "tensor(bfloat16)": (torch.bfloat16, None, None, 2),  # bfloat16 not supported by DlPack!
    "tensor(float)": (torch.float32, np.float32, C.c_float, 4),
    "tensor(double)": (torch.float64, np.float64, C.c_double, 8),
}


def to_pytorch(ort_tensor: OrtValue, clone_tensor: bool) -> torch.Tensor:
    """
    Convert OrtValue output by Onnx Runtime to Pytorch tensor.
    The process can be done in a zero copy way (depending of clone parameter).
    :param ort_tensor: output from Onnx Runtime
    :param clone_tensor: Onnx Runtime owns the storage array and will write on the next inference.
        By cloning you guarantee that the data won't change.
    :return: Pytorch tensor
    """
    ort_type = ort_tensor.data_type()
    torch_type, np_type, c_type, element_size = ort_conversion_table[ort_type]
    use_cuda = ort_tensor.device_name().lower() == "cuda"
    use_intermediate_dtype = np_type is None or (c_type is None and not use_cuda)
    # some types are not supported by numpy (like bfloat16), so we use intermediate dtype
    # same for ctype if tensor is on CPU
    if use_intermediate_dtype:
        np_type = np.byte
        # fake type as some don't exist in C, like float16
        c_type = C.c_byte
        nb_elements = np.prod(ort_tensor.shape())
        data_size = nb_elements * element_size
        input_shape = (data_size,)
    else:
        input_shape = ort_tensor.shape()
    if use_cuda:
        fake_owner = 1
        # size not used anywhere, so just put 0
        memory = cp.cuda.UnownedMemory(ort_tensor.data_ptr(), 0, fake_owner)
        memory_ptr = cp.cuda.MemoryPointer(memory, 0)
        # make sure interpret the array shape/dtype/strides correctly
        cp_array = cp.ndarray(shape=input_shape, memptr=memory_ptr, dtype=np_type)
        # cloning required otherwise ORT will recycle the storage array and put new values into it if new inf is done.
        torch_tensor: torch.Tensor = torch.from_dlpack(cp_array.toDlpack())
    else:
        data_pointer = C.cast(ort_tensor.data_ptr(), C.POINTER(c_type))
        np_tensor = np.ctypeslib.as_array(data_pointer, shape=input_shape)
        torch_tensor = torch.from_numpy(np_tensor)
    # convert back to the right type
    if use_intermediate_dtype:
        # https://github.com/csarofeen/pytorch/pull/1481 -> no casting, just reinterpret_cast
        torch_tensor = torch_tensor.view(torch_type)
        torch_tensor = torch_tensor.reshape(ort_tensor.shape())
    if clone_tensor:
        torch_tensor = torch_tensor.clone()
    return torch_tensor


def inference_onnx_binding(
    model_onnx: InferenceSession,
    inputs: Dict[str, torch.Tensor],
    device: str,
    device_id: int = 0,
    binding: Optional[IOBinding] = None,
    clone_tensor: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    Performs inference on ONNX Runtime in an optimized way.
    In particular, it avoids any Onnx Runtime output tensor copy.
    It means that Onnx Runtime is still owner of the array, and it will overwrite its content if you do another
    inference. To avoid any issue, just set clone_tensor to True (default).
    For best performance and lowest memory footprint, if you know what you are doing, set clone_tensor to False.

    :param model_onnx: ONNX model
    :param inputs: input torch tensor
    :param device: where to run the inference. One of [cpu, cuda]
    :param device_id: ID of the device where to run the inference, to be used when there are multiple GPUs, etc.
    :param binding: previously generated binding IO, will be reset.
    :param clone_tensor: clone Pytorch tensor to avoid its content being overwritten by Onnx Runtime
        at the next inference call.
    :return: a dict {axis name: output tensor}
    """
    assert isinstance(device, str)
    assert device in ["cpu", "cuda"], f"unexpected inference device: '{device}'"
    if binding is None:
        binding: IOBinding = model_onnx.io_binding()
    else:
        binding.clear_binding_inputs()
        binding.clear_binding_outputs()
    for input_onnx in model_onnx.get_inputs():
        if input_onnx.name not in inputs:  # some inputs may be optional
            continue
        tensor: torch.Tensor = inputs[input_onnx.name]
        tensor = tensor.detach()
        tensor = tensor.contiguous()
        binding.bind_input(
            name=input_onnx.name,
            device_type=device,
            device_id=device_id,
            element_type=torch_to_numpy_dtype_dict[tensor.dtype],
            shape=tuple(tensor.shape),
            buffer_ptr=tensor.data_ptr(),
        )
        inputs[input_onnx.name] = tensor

    for out in model_onnx.get_outputs():
        binding.bind_output(
            name=out.name,
            device_type=device,
            device_id=device_id,
        )
    binding.synchronize_inputs()
    model_onnx.run_with_iobinding(binding)
    binding.synchronize_outputs()
    outputs = dict()
    assert len(model_onnx.get_outputs()) == len(
        binding.get_outputs()
    ), f"{len(model_onnx.get_outputs())} != {len(binding.get_outputs())}"
    for out, t in zip(model_onnx.get_outputs(), binding.get_outputs()):
        outputs[out.name] = to_pytorch(t, clone_tensor=clone_tensor)
    return outputs
