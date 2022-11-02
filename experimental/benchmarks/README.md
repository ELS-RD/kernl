# Benchmark of Third-Party Libraries

This directory contains benchmarks of third-party libraries. The benchmarks are
written as simple Python script and have been run on a Nvidia 3090 RTX GPU, 128 Gb RAM, 12 cores Intel CPU.

Measures are done in wall-clock time, and output tensors are kept on GPU.

## [TensorRT](https://github.com/NVIDIA/TensorRT/)

### Version

We rely on the Docker image `nvcr.io/nvidia/tensorrt:22.09-py3`.

### Results

| batch | sequence length | Time (ms) |
|-------|-----------------|-----------|
| 1     | 16              | 0.0010    |
| 1     | 32              | 0.0010    |
| 1     | 64              | 0.0011    |
| 1     | 128             | 0.0013    |
| 1     | 256             | 0.0016    |
| 1     | 384             | 0.0026    |
| 1     | 512             | 0.0026    |
| 8     | 16              | 0.0011    |
| 8     | 32              | 0.0015    |
| 8     | 64              | 0.0019    |
| 8     | 128             | 0.0036    |
| 8     | 256             | 0.0064    |
| 8     | 384             | 0.0139    |
| 8     | 512             | 0.0139    |
| 32    | 16              | 0.0020    |
| 32    | 32              | 0.0031    |
| 32    | 64              | 0.0054    |
| 32    | 128             | 0.0103    |
| 32    | 256             | 0.0210    |

### Running the benchmark

```bash
docker run --rm -it --gpus all -v $(pwd):/work nvcr.io/nvidia/tensorrt:22.09-py3
cd /work
pip install transformers torch -U --extra-index-url https://download.pytorch.org/whl/cu116
python experimental/benchmarks/tensorrt_.py
```

### Notes

As `TensorRT` (`Myelin` code generator) complains where there are 2 dynamic axes, we build one model per batch size.
Only sequence length axis is dynamic.

Model building takes time, around 30mn on a beefy machine.

Most of the code has been taken from [transformer-deploy](https://github.com/ELS-RD/transformer-deploy).

It is important to note that `TensorRT` is a black box and we are not aware of simple ways to disable fast GELU, 
fp16 accumulation, or whatever optimization leveraged (we know they use many of them because of the precision issues 
we experienced in prod with this tool).

## [AITemplate](https://github.com/facebookincubator/AITemplate/)

### Version

Main branch commit used:

```shell
❯ git worktree list
# /mnt/workspace/AITemplate  445a20e [main]
```

### Results

| batch | sequence length | Time (ms) |
|-------|-----------------|-----------|
| 1     | 16              | 0.0012    |
| 1     | 32              | 0.0012    |
| 1     | 64              | 0.0013    |
| 1     | 128             | 0.0015    |
| 1     | 256             | 0.0020    |
| 1     | 384             | 0.0022    |
| 1     | 512             | 0.0031    |
| 8     | 16              | 0.0013    |
| 8     | 32              | 0.0017    |
| 8     | 64              | 0.0026    |
| 8     | 128             | 0.0043    |
| 8     | 256             | 0.0076    |
| 8     | 384             | 0.0115    |
| 8     | 512             | 0.0149    |
| 32    | 16              | 0.0026    |
| 32    | 32              | 0.0043    |
| 32    | 64              | 0.0073    |
| 32    | 128             | 0.0127    |
| 32    | 256             | 0.0242    |

### Running the benchmark

```shell
# in AITemplate root directory
./docker/build.sh cuda
# in kernl root directory
docker run --rm -it --gpus all -v $(pwd):/work -v $(pwd):/work ait
cp /work/experimental/benchmarks/aitemplate.py /AITemplate/examples/03_bert/demo_new.py
cd AITemplate/
python3 ./examples/03_bert/demo_new.py
# wait ...
cat measures.txt
```

### Notes

The script is based on the official [demo script](https://github.com/facebookincubator/AITemplate/tree/main/examples/03_bert).

The model do not support `attention_mask`, so we don't use it in benchmarks.
It is important to keep in mind that `attention mask` adds operations on top of an already computation bounded kernel,
in particular for long sequence length inputs.
Said otherwise, it would likely make it slower.
Moreover, without `attention mask`, batch inference is useless right now.  
There is a multithreads mode which would get much more overhead than batch mode (launching n threads times more kernels).

**TL;DR**: numbers for AITemplate have to be taken with a grain of salt.

An issue has been opened [here](https://github.com/facebookincubator/AITemplate/issues/46) on the repo:

```cite
@antinucleon
The current BERT example is only used for benchmarking purposes on fixed
length without mask.

We are currently working with CUTLASS team on a grouped Attention
optimization, which will remove paddings & mask for dynamic sequences. It
will appear in next CUTLASS & AIT release.
```

We choose to use the following options:

* accumulation in `FP32` (instead of default `FP16`): 
  `FP16` accumulation reduces output precision, and has been a source of issues in our experience with other tools.
* `GELU` instead of `fast GELU`:
  fast GELU is basically a simple operation with hard-coded values. It's faster but reduces output precision.
  FWIW, `Kernl` also have support of fast GELU but it is disabled by default.
* `CUDA graphs` enabled: this technology remove kernel launching overhead, and is a good practice to use it when possible.

We do not use the [benchmark](https://github.com/facebookincubator/AITemplate/blob/main/examples/03_bert/benchmark_ait.py)
script provided in the `AITemplate` repo because:
* it reports GPU times through CUDA events and we compare inference engines on wall-clock times which better matches our end-to-end use cases.
* we are not sure of the meaning of the reported times in case of multiple threads/cuda streams used
  (see [here](https://github.com/facebookincubator/AITemplate/issues/44)), it doesn't match latency or throughput definition

CPP implementation of benchmark function is [here](https://github.com/facebookincubator/AITemplate/blob/44026ba7e7f5376a80cf0f2b333a0f25c0eeda6c/static/csrc/model_container.cpp).


## [TorchDynamo + inductor](https://github.com/pytorch/torchdynamo)

### Version

* Triton: https://github.com/openai/triton@af76c989eb4799b015f8b288ccd8421558772e56#subdirectory=python
* PyTorch (includes TorchDynamo): 1.14.0.dev20221015+cu116

### Results

| batch | sequence length | Time (ms) |
|-------|-----------------|-----------|
| 1     | 16              | 0.0018    |
| 1     | 32              | 0.0020    |
| 1     | 64              | 0.0020    |
| 1     | 128             | 0.0025    |
| 1     | 256             | 0.0030    |
| 1     | 384             | 0.0036    |
| 1     | 512             | 0.0048    |
| 8     | 16              | 0.0023    |
| 8     | 32              | 0.0027    |
| 8     | 64              | 0.0039    |
| 8     | 128             | 0.0065    |
| 8     | 256             | 0.0117    |
| 8     | 386             | 0.0156    |
| 8     | 512             | 0.0212    |
| 32    | 16              | 0.0039    |
| 32    | 32              | 0.0062    |
| 32    | 64              | 0.0108    |
| 32    | 128             | 0.0177    |
| 32    | 256             | 0.0357    |

### Running the benchmark

```shell
# reuse AITemplate docker image based on nvidia/cuda:11.6.2-devel-ubuntu20.04
docker run --rm -it --gpus all -v $(pwd):/work -v $(pwd):/work ait
cd work/
apt update
apt install git
pip3 install --pre torch==1.14.0.dev20221015+cu116 --extra-index-url https://download.pytorch.org/whl/nightly/cu116 -U
pip3 install git+https://github.com/openai/triton@af76c989eb4799b015f8b288ccd8421558772e56#subdirectory=python
python3 experimental/benchmarks/inductor.py
```

### Notes

`Torchinductor` is still in prototype stage, results may be different with final version.  
We are using the version included in PyTorch nightly for this benchmark.  
The dependency of this project is an older version not requiring nightly version as we only need `TorchDynamo`.
Project info on https://github.com/pytorch/torchdynamo even if code is not anymore updated in this repo.

We tried several disabled by default optimizations but none of them worked:

* `config.aggressive_fusion = True`: significantly slower when enabled on our GPU.
* `config.inplace_buffers = True`: crash, see https://github.com/pytorch/torchdynamo/issues/823
* `config.triton.mm = "triton"` (same for `"autotune"`): crash, even when trying with `config.triton.autotune = False`

By default, `CUDA graphs` is enabled.

The last one is important and should bring some speedup when working.

## [Deepspeed](https://github.com/microsoft/DeepSpeed)

### Version

0.7.4

### Results

| batch | sequence length | Time (ms) |
|-------|-----------------|-----------|
| 1     | 16              | 0.0009    |
| 1     | 32              | 0.0008    |
| 1     | 64              | 0.0009    |
| 1     | 128             | 0.0012    |
| 1     | 256             | 0.0019    |
| 1     | 384             | 0.0023    |
| 1     | 512             | 0.0032    |
| 8     | 16              | 0.0011    |
| 8     | 32              | 0.0016    |
| 8     | 64              | 0.0025    |
| 8     | 128             | 0.0051    |
| 8     | 256             | 0.0106    |
| 8     | 384             | 0.0161    |
| 8     | 512             | 0.0219    |
| 32    | 16              | 0.0025    |
| 32    | 32              | 0.0050    |
| 32    | 64              | 0.0097    |
| 32    | 128             | 0.0176    |
| 32    | 256             | 0.0374    |

### Running the benchmark

```shell
# reuse AITemplate docker image based on nvidia/cuda:11.6.2-devel-ubuntu20.04
docker run --rm -it --gpus all -v $(pwd):/work -v $(pwd):/work ait
pip install deepspeed transformers
deepspeed --num_gpus 1 experimental/benchmarks/deepspeed_.py --deepspeed
```

### Notes

The benchmark script is built over the 
[one](https://github.com/microsoft/DeepSpeed/blob/master/benchmarks/inference/bert-bench.py) provided in the 
`deepspeed` repo.

We rebuilt a model for each shape to leverage CUDA Graphs and get best possible performances.  
In real scenario, we would need something on top of it to handle multiple graphs.

Model got its weights completly converted to fp16 (instead of doing mixed precision) as it is done that way in the 
benchmark script. 

## [ONNX Runtime](https://github.com/microsoft/onnxruntime)

### Version

1.12.1

### Results

| batch | sequence length | Time (ms) |
|-------|-----------------|-----------|
| 1     | 16              | 0.0024    |
| 1     | 128             | 0.0025    |
| 1     | 256             | 0.0027    |
| 1     | 384             | 0.0027    |
| 1     | 512             | 0.0038    |
| 8     | 16              | 0.0025    |
| 8     | 128             | 0.0057    |
| 8     | 256             | 0.0112    |
| 8     | 384             | 0.0155    |
| 8     | 512             | 0.0207    |
| 32    | 16              | 0.0031    |
| 32    | 128             | 0.0177    |
| 32    | 256             | 0.0348    |

### Running the benchmark

```shell
python experimental/benchmarks/onnxrt.py
```

### Notes

We have not been able to use CUDA graphs on this engine.
It appears to have a limited support:
https://github.com/microsoft/onnxruntime/issues/12977#issuecomment-1258406358

It may explain why we are not able to get the same performances as the other engines on small shapes where overhead
dominates.

We have converted the model to fp16 to get best performances using utils from ONNX Runtime package.
