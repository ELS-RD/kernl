import dataclasses
import logging
import os.path
from dataclasses import dataclass
from time import time
from typing import Callable, Dict, List, Optional

import tensorrt as trt
import torch
from tensorrt import ICudaEngine, IExecutionContext, ILayer, INetworkDefinition, Logger, Runtime
from tensorrt.tensorrt import Builder, IBuilderConfig, IElementWiseLayer, IOptimizationProfile, IReduceLayer, OnnxParser


@dataclass
class TensorRTShape:

    min_shape: List[int]
    optimal_shape: List[int]
    max_shape: List[int]
    input_name: Optional[str]

    def check_validity(self) -> None:

        assert len(self.min_shape) == len(self.optimal_shape) == len(self.max_shape)
        assert len(self.min_shape) > 0
        assert self.min_shape[0] > 0 and self.optimal_shape[0] > 0 and self.max_shape[0] > 0
        assert self.input_name is not None

    def make_copy(self, input_name: str) -> "TensorRTShape":

        instance_copy = dataclasses.replace(self)
        instance_copy.input_name = input_name
        return instance_copy

    def generate_multiple_shapes(self, input_names: List[str]) -> List["TensorRTShape"]:

        assert self.input_name is None, f"input name is not None: {self.input_name}"
        result = list()
        for name in input_names:
            shape = self.make_copy(input_name=name)
            result.append(shape)
        return result


def fix_fp16_network(network_definition: INetworkDefinition) -> INetworkDefinition:
    # search for patterns which may overflow in FP16 precision, we force FP32 precisions for those nodes
    for layer_index in range(network_definition.num_layers - 1):
        layer: ILayer = network_definition.get_layer(layer_index)
        next_layer: ILayer = network_definition.get_layer(layer_index + 1)
        # POW operation usually followed by mean reduce
        if layer.type == trt.LayerType.ELEMENTWISE and next_layer.type == trt.LayerType.REDUCE:
            # casting to get access to op attribute
            layer.__class__ = IElementWiseLayer
            next_layer.__class__ = IReduceLayer
            if layer.op == trt.ElementWiseOperation.POW:
                layer.precision = trt.DataType.FLOAT
                next_layer.precision = trt.DataType.FLOAT
            layer.set_output_type(index=0, dtype=trt.DataType.FLOAT)
            next_layer.set_output_type(index=0, dtype=trt.DataType.FLOAT)
    return network_definition


def build_engine(
    runtime: Runtime,
    onnx_file_path: str,
    logger: Logger,
    fp16: bool,
    int8: bool,
    workspace_size: Optional[int] = None,
    fp16_fix: Callable[[INetworkDefinition], INetworkDefinition] = fix_fp16_network,
    **kwargs,
) -> ICudaEngine:
    # default input shape
    if "min_shape" in kwargs and "optimal_shape" in kwargs and "max_shape" in kwargs:
        default_shape = TensorRTShape(
            min_shape=kwargs["min_shape"],
            optimal_shape=kwargs["optimal_shape"],
            max_shape=kwargs["max_shape"],
            input_name=None,
        )
        input_shapes = [default_shape]
    else:
        assert "input_shapes" in kwargs, "missing input shapes"
        input_shapes: List[TensorRTShape] = kwargs["input_shapes"]

    builder: Builder = trt.Builder(logger)
    network_def: INetworkDefinition = builder.create_network(
        flags=1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser: OnnxParser = trt.OnnxParser(network_def, logger)
    config: IBuilderConfig = builder.create_builder_config()
    if workspace_size is not None:
        config.set_memory_pool_limit(trt.tensorrt.MemoryPoolType.DLA_GLOBAL_DRAM, workspace_size)
    config.set_tactic_sources(
        tactic_sources=1 << int(trt.TacticSource.CUBLAS)
        | 1 << int(trt.TacticSource.CUBLAS_LT)
        | 1 << int(trt.TacticSource.CUDNN)  # trt advised to use cuDNN for transfo architecture
    )
    if int8:
        config.set_flag(trt.BuilderFlag.INT8)
    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    config.set_flag(trt.BuilderFlag.DISABLE_TIMING_CACHE)
    # https://github.com/NVIDIA/TensorRT/issues/1196 (sometimes big diff in output when using FP16)
    config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
    logger.log(msg="parsing TensorRT model", severity=trt.ILogger.INFO)
    with open(onnx_file_path, "rb") as f:
        # file path needed for models with external dataformat
        # https://github.com/onnx/onnx-tensorrt/issues/818
        parser.parse(model=f.read(), path=onnx_file_path)
    profile: IOptimizationProfile = builder.create_optimization_profile()
    # duplicate default shape (one for each input)
    if len(input_shapes) == 1 and input_shapes[0].input_name is None:
        names = [network_def.get_input(num_input).name for num_input in range(network_def.num_inputs)]
        input_shapes = input_shapes[0].generate_multiple_shapes(input_names=names)

    for shape in input_shapes:
        shape.check_validity()
        profile.set_shape(
            input=shape.input_name,
            min=shape.min_shape,
            opt=shape.optimal_shape,
            max=shape.max_shape,
        )
    if "shape_tensors" in kwargs:
        for shape in kwargs["shape_tensors"]:
            profile.set_shape_input(
                input=shape.input_name,
                min=shape.min_shape,
                opt=shape.optimal_shape,
                max=shape.max_shape,
            )
    config.add_optimization_profile(profile)
    if fp16:
        network_def = fp16_fix(network_def)

    logger.log(msg="building engine. depending on model size this may take a while", severity=trt.ILogger.WARNING)
    start = time()
    trt_engine = builder.build_serialized_network(network_def, config)
    engine: ICudaEngine = runtime.deserialize_cuda_engine(trt_engine)
    logger.log(msg=f"building engine took {time() - start:4.1f} seconds", severity=trt.ILogger.WARNING)
    assert engine is not None, "error during engine generation, check error messages above :-("
    return engine


def get_output_tensors(
    context: trt.IExecutionContext,
    host_inputs: List[torch.Tensor],
    input_binding_idxs: List[int],
    output_binding_idxs: List[int],
) -> Dict[str, torch.Tensor]:
    # explicitly set dynamic input shapes, so dynamic output shapes can be computed internally
    for host_input, binding_index in zip(host_inputs, input_binding_idxs):
        context.set_binding_shape(binding_index, tuple(host_input.shape))
    # assert context.all_binding_shapes_specified
    device_outputs: Dict[str, torch.Tensor] = dict()
    for binding_index in output_binding_idxs:
        # TensorRT computes output shape based on input shape provided above
        output_shape = context.get_binding_shape(binding=binding_index)
        output_name = context.engine.get_binding_name(index=binding_index)
        # allocate buffers to hold output results
        device_outputs[output_name] = torch.empty(tuple(output_shape), device="cuda")
    return device_outputs


def infer_tensorrt(
    context: IExecutionContext,
    inputs: Dict[str, torch.Tensor],
    input_binding_idxs: List[int],
    output_binding_idxs: List[int],
) -> Dict[str, torch.Tensor]:

    input_tensors: List[torch.Tensor] = list()
    for i in range(context.engine.num_bindings):
        if not context.engine.binding_is_input(index=i):
            continue
        tensor_name = context.engine.get_binding_name(i)
        assert tensor_name in inputs, f"input not provided: {tensor_name}"
        tensor = inputs[tensor_name]
        assert isinstance(tensor, torch.Tensor), f"unexpected tensor class: {type(tensor)}"
        assert tensor.device.type == "cuda", f"unexpected device type (trt only works on CUDA): {tensor.device.type}"
        # warning: small changes in output if int64 is used instead of int32
        if tensor.dtype in [torch.int64, torch.long]:
            logging.warning(f"using {tensor.dtype} instead of int32 for {tensor_name}, will be casted to int32")
            tensor = tensor.type(torch.int32)
        input_tensors.append(tensor)
    # calculate input shape, bind it, allocate GPU memory for the output
    outputs: Dict[str, torch.Tensor] = get_output_tensors(
        context, input_tensors, input_binding_idxs, output_binding_idxs
    )
    bindings = [int(i.data_ptr()) for i in input_tensors + list(outputs.values())]
    assert context.execute_async_v2(
        bindings, torch.cuda.current_stream().cuda_stream
    ), "failure during execution of inference"
    torch.cuda.current_stream().synchronize()  # sync all CUDA ops

    return outputs


def load_engine(
    runtime: Runtime, engine_file_path: str, profile_index: int = 0
) -> Callable[[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]:
    with open(file=engine_file_path, mode="rb") as f:
        engine: ICudaEngine = runtime.deserialize_cuda_engine(f.read())
        stream: int = torch.cuda.current_stream().cuda_stream
        context: IExecutionContext = engine.create_execution_context()
        context.set_optimization_profile_async(profile_index=profile_index, stream_handle=stream)
        # retrieve input/output IDs
        input_binding_idxs, output_binding_idxs = get_binding_idxs(engine, profile_index)  # type: List[int], List[int]

        def tensorrt_model(inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
            return infer_tensorrt(
                context=context,
                inputs=inputs,
                input_binding_idxs=input_binding_idxs,
                output_binding_idxs=output_binding_idxs,
            )

        return tensorrt_model


def get_binding_idxs(engine: trt.ICudaEngine, profile_index: int):
    num_bindings_per_profile = engine.num_bindings // engine.num_optimization_profiles
    start_binding = profile_index * num_bindings_per_profile
    end_binding = start_binding + num_bindings_per_profile  # Separate input and output binding indices for convenience
    input_binding_idxs: List[int] = []
    output_binding_idxs: List[int] = []
    for binding_index in range(start_binding, end_binding):
        if engine.binding_is_input(binding_index):
            input_binding_idxs.append(binding_index)
        else:
            output_binding_idxs.append(binding_index)
    return input_binding_idxs, output_binding_idxs


def build_model_tensorrt(model_name: str, model_dir: str):
    import tensorrt as trt
    from tensorrt.tensorrt import Logger, Runtime, ICudaEngine
    from test.models.onnx_utils import get_model_onnx

    trt_model_name = f"{model_name}.plan"
    trt_model_path = os.path.join(model_dir, trt_model_name)
    trt_logger: Logger = trt.Logger(trt.Logger.ERROR)
    runtime: Runtime = trt.Runtime(trt_logger)
    if os.path.exists(os.path.join(model_dir, trt_model_name)):
        model_trt = load_engine(runtime=runtime, engine_file_path=trt_model_path)
        return model_trt
    if not os.path.exists(os.path.join(model_dir, f"{model_name}.onnx")):
        _ = get_model_onnx(model_name, model_dir)
    input_id_shape = TensorRTShape(
        min_shape=[1, 16], optimal_shape=[16, 512], max_shape=[16, 512], input_name="input_ids"
    )
    attention_mask_shape = TensorRTShape(
        min_shape=[1, 16], optimal_shape=[16, 512], max_shape=[16, 512], input_name="attention_mask"
    )
    token_type_id_shape = TensorRTShape(
        min_shape=[1, 16], optimal_shape=[16, 512], max_shape=[16, 512], input_name="token_type_ids"
    )
    input_shapes = [input_id_shape, attention_mask_shape, token_type_id_shape]
    output_shape = TensorRTShape(
        min_shape=[1],
        optimal_shape=[1],
        max_shape=[1],
        input_name="last_hidden_state",
    )
    shape_tensors = [output_shape]
    engine: ICudaEngine = build_engine(
        runtime=runtime,
        onnx_file_path=os.path.join(model_dir, f"{model_name}.onnx"),
        logger=trt_logger,
        workspace_size=20000 * 1024 ** 2,
        fp16=False,
        int8=False,
        input_shapes=input_shapes,
        shape_tensors=shape_tensors,
    )
    # save engine:
    with open(trt_model_name, "wb") as f:
        f.write(engine.serialize())
    model_trt = load_engine(runtime=runtime, engine_file_path=trt_model_name)
    return model_trt


def get_trt_model(model_name: str, model_dir: str):
    from transformers.modeling_outputs import BaseModelOutputWithPooling

    model_trt = build_model_tensorrt(model_name, model_dir)

    def run(*args, **kwargs):
        inputs = {
            "input_ids": kwargs["input_ids"],
            "attention_mask": kwargs["attention_mask"],
            "token_type_ids": kwargs["token_type_ids"]
        }
        outputs = model_trt(inputs=inputs)
        return BaseModelOutputWithPooling(
            last_hidden_state=outputs["last_hidden_state"],
            pooler_output=outputs["1607"]
        )
    return run
