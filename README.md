# triton-xp

Optimized kernels for `transformer` models.

## Install dependencies

```shell
pip install -r requirements.txt
```

## Test and Benchmark

### Test

```shell
pytest
```

### Compare benchmarks

You can group benchmark results to compare them

```shell
pytest --benchmark-group-by fullfunc,param:batch,param:size
```