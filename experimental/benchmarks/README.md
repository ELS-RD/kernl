# Benchmark of Third-Party Libraries

This directory contains benchmarks of third-party libraries. The benchmarks are
written as simple Python script and have been run on a Nvidia 3090 RTX GPU.

## [TensorRT](https://github.com/NVIDIA/TensorRT/)

### Version

We rely on the Docker image `nvcr.io/nvidia/tensorrt:22.09-py3`.

### Results

| batch | sequence length | Time (ms) |
|-------|-----------------|-----------|
 | 1     | 16              | 0.0010    |
 | 1     | 32              | 0.0012    |
 | 1     | 64              | 0.0011    |
 | 1     | 128             | 0.0013    |
 | 1     | 256             | 0.0016    |
 | 1     | 512             | 0.0026    |
 | 8     | 16              | 0.0012    |
 | 8     | 32              | 0.0015    |
 | 8     | 64              | 0.0020    |
 | 8     | 128             | 0.0036    |
 | 8     | 256             | 0.0064    |
 | 8     | 512             | 0.0142    |
 | 32    | 16              | 0.0020    |
 | 32    | 32              | 0.0031    |
 | 32    | 64              | 0.0055    |
 | 32    | 128             | 0.0103    |
 | 32    | 256             | 0.0210    |

### Running the benchmark

```bash
docker run --rm -it --gpus all -v $(pwd):/work nvcr.io/nvidia/tensorrt:22.09-py3
cd /work
pip install transformers torch -U --extra-index-url https://download.pytorch.org/whl/cu116
python experimental/benchmarks/tensorrt.py
```

### Notes

As `TensorRT` complains where there are 2 dynamic axes, we build one model per batch size.
Only sequence length axis is dynamic.

Model building takes time, around 30mn on a beefy machine.

Most of the code has been taken from [transformer-deploy](https://github.com/ELS-RD/transformer-deploy).

It is important to note that `TensorRT` is a black box and we cannot disable fast GELU, fp16 accumulation, 
or whatever optimization they are using.

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
| 1     | 512             | 0.0031    |
| 8     | 16              | 0.0013    |
| 8     | 32              | 0.0017    |
| 8     | 64              | 0.0026    |
| 8     | 128             | 0.0044    |
| 8     | 256             | 0.0076    |
| 8     | 512             | 0.0149    |
| 32    | 16              | 0.0027    |
| 32    | 32              | 0.0043    |
| 32    | 64              | 0.0073    |
| 32    | 128             | 0.0131    |
| 32    | 256             | 0.0249    |

### Running the benchmark

```shell
# in AITemplate root directory
./docker/build.sh cuda
# in kernl root directory
docker run --rm -it  --gpus all -v $(pwd):/work -v $(pwd):/work ait
cp /work/experimental/benchmarks/aitemplate.py /AITemplate/examples/03_bert/demo_new.py
cd AITemplate/
python3 ./examples/03_bert/demo_new.py
# wait ...
cat measures.txt
```

### Notes

The script is based on the official [demo script](https://github.com/facebookincubator/AITemplate/tree/main/examples/03_bert).

We choose to use the following options:

* accumulation in `FP32` (instead of default `FP16`): 
  `FP16` accumulation reduces output precision, and has been a source of issues in our experience with other tools.
* `GELU` instead of `fast GELU`:
  fast GELU is basically a simple operation with hard-coded values. It's faster but reduces output precision.
  FWIW, `Kernl` also have support of fast GELU but it is disabled by default.
* `CUDA graphs` enabled: this technology remove kernel launching overhead, and is a good practice to use it when possible.

We don't use the [benchmark](https://github.com/facebookincubator/AITemplate/blob/main/examples/03_bert/benchmark_ait.py)
script provided in the `AITemplate` repo because it reports GPU times through CUDA events and we want to measure the wall 
clock time which better match our end to end case.

Differences are mostly on short input shapes where CPU overhead dominates. 

CPP implementation of benchmark function is [here](https://github.com/facebookincubator/AITemplate/blob/44026ba7e7f5376a80cf0f2b333a0f25c0eeda6c/static/csrc/model_container.cpp).
