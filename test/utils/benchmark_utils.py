import logging
import os

from transformers import AutoConfig, PretrainedConfig

from test.utils.ort_utils import inference_onnx_binding

logger = logging.getLogger(__name__)


def get_model_onnx(model_name: str, model_dir_path: str):
    try:
        from optimum.onnxruntime import ORTModelForSequenceClassification
        from transformer_deploy.backends.ort_utils import create_model_for_provider, optimize_onnx
        from transformer_deploy.backends.pytorch_utils import get_model_size
    except ImportError:
        logger.error(
            "It seems that onnx runtime is not yet installed. Onnx models will not be included in the benchmark."
        )
        return
    model_onnx_name = f"{model_name}.onnx"
    model_file_path = os.path.join(model_dir_path, model_onnx_name)
    # Load a model from transformers and export it through the ONNX format
    model = ORTModelForSequenceClassification.from_pretrained(model_name, from_transformers=True)
    model.save_pretrained(model_file_path, file_name=model_onnx_name)

    def run(*args, **kwargs):
        return inference_onnx_binding(model_onnx=model, device="cuda", *args, **kwargs)

    return run


def get_model_onnx_optimized(model_name: str, model_dir_path: str):
    try:
        from transformer_deploy.backends.ort_utils import create_model_for_provider, optimize_onnx
        from transformer_deploy.backends.pytorch_utils import get_model_size
    except ImportError:
        logger.error(
            "It seems that the transformer-deploy library is not yet installed. "
            "Onnx models will not be included in the benchmark."
        )
        return
    _ = get_model_onnx(model_name, model_dir_path)
    model_onnx_name = f"{model_name}.onnx"
    onnx_optim_model_path = os.path.join(model_dir_path, f"{model_name}_optim.onnx")
    num_attention_heads, hidden_size = get_model_size(path=model_name)
    model_config: PretrainedConfig = AutoConfig.from_pretrained(
        pretrained_model_name_or_path=model_name
    )
    onnx_model_path = os.path.join(model_dir_path, model_onnx_name)
    optimize_onnx(
        onnx_path=onnx_model_path,
        onnx_optim_model_path=onnx_optim_model_path,
        fp16=True,
        use_cuda=True,
        num_attention_heads=num_attention_heads,
        hidden_size=hidden_size,
        architecture=model_config.model_type,
    )
    model_onnx_optim = create_model_for_provider(path=onnx_optim_model_path, provider_to_use="CUDAExecutionProvider")

    def run(*args, **kwargs):
        return inference_onnx_binding(model_onnx=model_onnx_optim, device="cuda", *args, **kwargs)

    return run


def get_model_tensorrt(model_name: str, model_dir_path: str, optimized_onnx: bool = False):
    try:
        from tensorrt.tensorrt import Logger
        import tensorrt as trt
        from tensorrt import Runtime
        from tensorrt.tensorrt import ICudaEngine
        from test.utils.trt_utils import build_engine, save_engine, load_engine
    except ImportError:
        logger.error(
            "It seems that TensorRT is not yet installed. It is required to include TensorRT in benchmark."
            "Please find installation instruction on: "
            "https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html"
        )
        return
    _ = get_model_onnx(model_name, model_dir_path)
    trt_logger: Logger = trt.Logger(trt.Logger.ERROR)
    runtime: Runtime = trt.Runtime(trt_logger)
    trt_model_name = f"{model_name}.plan"
    model_onnx_name = f"{model_name}_optim.onnx" if optimized_onnx else f"{model_name}.onnx"
    engine: ICudaEngine = build_engine(
        runtime=runtime,
        onnx_file_path=f"{model_onnx_name}/{model_onnx_name}",
        logger=trt_logger,
        workspace_size=20000 * 1024 ** 2,
        fp16=False,  # for tests only
        int8=False,
        min_shape=(1, 256),
        optimal_shape=(16, 256),
        max_shape=(16, 256)
    )
    save_engine(engine, trt_model_name)
    model_trt = load_engine(runtime=runtime, engine_file_path=trt_model_name)
    return model_trt
