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

The model do not support `attention_mask`, so we don't use it in benchmarks.
It is important to keep in mind that `attention mask` adds operations on top of an already computation bounded kernel.
Said otherwise, it would it slower.
On the other side, without `attentino mask`, batch inference is useless right now.
So numbers for AITemplate have to be taken with a grain of salt.

According 

For follow-up, an issue has been opened [here](https://github.com/facebookincubator/AITemplate/issues/46) on the repo:

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
* it reports GPU times through CUDA events and compare inference engines on wall-clock times which better matches our end-to-end use cases.
* we are not sure of the meaning of the reported times in case of multiple threads/cuda streams used
  (see [here](https://github.com/facebookincubator/AITemplate/issues/44)), it doesn't match latency or throughput definition

CPP implementation of benchmark function is [here](https://github.com/facebookincubator/AITemplate/blob/44026ba7e7f5376a80cf0f2b333a0f25c0eeda6c/static/csrc/model_container.cpp).


## [Torchdynamo + inductor](https://github.com/pytorch/torchdynamo)

### Version

* Triton: https://github.com/openai/triton@af76c989eb4799b015f8b288ccd8421558772e56#subdirectory=python
* Pytorch (includes Torchdynamo): 1.14.0.dev20221015+cu117

### Results

| batch | sequence length | Time (ms) |
|-------|-----------------|-----------|
| 1     | 16              | 0.0017    |
| 1     | 32              | 0.0018    |
| 1     | 64              | 0.0019    |
| 1     | 128             | 0.0024    |
| 1     | 256             | 0.0029    |
| 1     | 512             | 0.0047    |
| 8     | 16              | 0.0024    |
| 8     | 32              | 0.0028    |
| 8     | 64              | 0.0042    |
| 8     | 128             | 0.0069    |
| 8     | 256             | 0.0115    |
| 8     | 512             | 0.0207    |
| 32    | 16              | 0.0041    |
| 32    | 32              | 0.0066    |
| 32    | 64              | 0.0114    |
| 32    | 128             | 0.0173    |
| 32    | 256             | 0.0345    |

### Running the benchmark

```shell
python experimental/benchmarks/inductor.py
```

### Notes

`Torchinductor` is still in prototype stage, results may be different with final version.  
We are using the version included in `Pytorch` nightly for this benchmark.  
The dependency of this project is an older version not requiring nightly version as we only need `Torchdynamo`.
Project info on https://github.com/pytorch/torchdynamo even if code is not anymore updated in this repo.

We tried several disabled by default optimizations but none of them worked:

* `config.aggressive_fusion = True`: significantly slower when enabled on our GPU.
* `config.inplace_buffers = True`: crash, see https://github.com/pytorch/torchdynamo/issues/823
* `config.triton.mm = "triton"` (same for `"autotune"`): crash, even when trying with `config.triton.autotune = False`

By default, `CUDA graphs` is enabled.

The last one is important and should bring some speedup when working.
