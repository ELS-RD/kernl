import logging
import os

logger = logging.getLogger(__name__)


def get_model_onnx(model_name: str, model_dir_path: str):
    try:
        from optimum.onnxruntime import ORTModelForFeatureExtraction
    except ImportError:
        logger.error(
            "It seems that onnx runtime is not yet installed. Onnx models will not be included in the benchmark."
        )
        return
    model_onnx_name = f"{model_name}.onnx"
    # Load a model from transformers and export it through the ONNX format
    model = ORTModelForFeatureExtraction.from_pretrained(model_name, from_transformers=True)
    model.save_pretrained(model_dir_path, file_name=model_onnx_name)

    def run(*args, **kwargs):
        outputs = model(
            input_ids=kwargs["input_ids"],
            attention_mask=kwargs["attention_mask"],
            token_type_ids=kwargs["token_type_ids"]
        )
        return outputs

    return run


def get_model_tensorrt(model_name: str, model_dir_path: str):
    try:
        from tensorrt.tensorrt import Logger
        import tensorrt as trt
        from tensorrt import Runtime
        from tensorrt.tensorrt import ICudaEngine
        from test.utils.trt_utils import build_engine, save_engine, load_engine, TensorRTShape
    except ImportError:
        logger.error(
            "It seems that TensorRT is not yet installed. It is required to include TensorRT in benchmark."
            "Please find installation instruction on: "
            "https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html"
        )
        return
    if not os.path.exists(os.path.join(model_dir_path, f"{model_name}.onnx")):
        _ = get_model_onnx(model_name, model_dir_path)
    trt_logger: Logger = trt.Logger(trt.Logger.ERROR)
    runtime: Runtime = trt.Runtime(trt_logger)
    trt_model_name = f"{model_name}.plan"
    model_onnx_name = f"{model_name}.onnx"
    input_id_shape = TensorRTShape(
        min_shape=[1, 512], optimal_shape=[32, 512], max_shape=[32, 512], input_name="input_ids"
    )
    attention_mask_shape = TensorRTShape(
        min_shape=[1, 512], optimal_shape=[32, 512], max_shape=[32, 512], input_name="attention_mask"
    )
    token_type_id_shape = TensorRTShape(
        min_shape=[1, 512], optimal_shape=[32, 512], max_shape=[32, 512], input_name="token_type_ids"
    )
    input_shapes = [input_id_shape, attention_mask_shape, token_type_id_shape]
    output_shape = TensorRTShape(
        min_shape=[1],
        optimal_shape=[1],
        max_shape=[1],
        input_name="last_hidden_state",
    )
    shape_tensors = [output_shape]
    engine_path = os.path.join(model_dir_path, f"{trt_model_name}")
    if not os.path.exists(engine_path):
        engine: ICudaEngine = build_engine(
            runtime=runtime,
            onnx_file_path=os.path.join(model_dir_path, model_onnx_name),
            logger=trt_logger,
            workspace_size=20000 * 1024 ** 2,
            fp16=False,
            int8=False,
            input_shapes=input_shapes,
            shape_tensors=shape_tensors,
        )
        save_engine(engine, engine_path)
    model_trt = load_engine(runtime=runtime, engine_file_path=engine_path)

    def run(*args, **kwargs):
        inputs = {
            "input_ids": kwargs["input_ids"].to("cuda"),
            "attention_mask": kwargs["attention_mask"].to("cuda"),
            "token_type_ids": kwargs["token_type_ids"].to("cuda")
        }
        return model_trt(inputs=inputs)

    return run
