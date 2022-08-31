# triton-xp

Optimized kernels for `transformer` models.

## Install dependencies

```shell
pip install -r requirements.txt
```

## Test and Benchmark

### Conventions

- A test function using benchmark features must have a name that starts with `test_benchmark_`
- Benchmark function must have a param called `implementation` when benchmarking the same operation using different
  strategy

### Run all

```shell
pytest
```

### Run benchmarks

```shell
pytest -k benchmark
```

### Run tests

```shell
pytest -k "not benchmark"
```

### Compare benchmarks

You can group benchmark results to compare them.  
Check that the field you are requesting is used in the benchmark you want.

```python

```shell
pytest
test / test_linear_layer.py - -benchmark - group - by
fullfunc, param: batch, param: size
# or for all tests (at the time of writing)
pytest - -benchmark - group - by
fullfunc, param: batch
```

### Execute benchmark visualization server

You must first run benchmarks

```shell
python3.8 server.py
```

## Create new patterns to replace fx graph nodes

The first step to replace function/module calls in the graph is to create the pattern that will be replaced.
The easiest way to do this is to [convert the model to a fx graph](https://pytorch.org/docs/stable/fx.html), and then
print it with `utils.graph_report` or by printing the code `print(you_graph_module.code)`

Then you can use [replace_pattern](https://pytorch.org/docs/stable/fx.html#torch.fx.replace_pattern) to replace the
pattern in the graph. We have our own version of `replace_pattern` with some enhancements to work with modules for
example. You can find examples of that in `optimizer` folder.