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

You can group benchmark results to compare them.  
Check that the field you are requesting is used in the benchmark you want.

```python

```shell
pytest test/test_linear_layer.py --benchmark-group-by fullfunc,param:batch,param:size
# or for all tests (at the time of writing)
pytest --benchmark-group-by fullfunc,param:batch
```