# nucle-ai

Optimized kernels for `transformer` models.

## Install dependencies

**IMPORTANT**: This package requires `pytorch` being installed.  
Please install it first.

```shell
pip install torch -U --extra-index-url https://download.pytorch.org/whl/cu116
git clone https://github.com/ELS-RD/nucle-ai
pip install -e .
# or to enable all benchmarks
pip install -e ".[benchmark]"
```

This project requires `Python` >= 3.9.

## Test and Benchmark

### Conventions

- A test function using benchmark features must have a name that starts with `test_benchmark_`
- Benchmark function must have a param called `implementation` when benchmarking the same operation using different
  strategy

### Run tests and benchmarks

```shell
# tada!
pytest
```

Some rules on how `PyTest` works, in particular for benchmarks:

- add `-k` to filter tests/benchmarks by their name like `pytest -k benchmark` to run only tests with `benchmark` 
 in their name
- you can combine expressions in the filter: `pytest -k "benchmark and not bert"` if you want to run all benchmarks 
  except those related to BERT
- to group and compare benchmark measures, use `pytest -k benchmark --benchmark-group-by ...`:
  - groupinng by names: `pytest -k benchmark --benchmark-group-by fullfunc`
  - grouping by names of parameters: `pytest -k benchmark --benchmark-group-by param:implementation,param:shape`
    - `param:x`, `x` is the parameter name in `@pytest.mark.parametrize`
  - combining both: `pytest -k benchmark --benchmark-group-by fullfunc,param:implementation`
- add `-s` to see the output of the tests (print, etc.)
- add `-v` to see the verbose output of the tests

*WARNING*: `param:X` will make PyTest crash if `X` is not a parameter of at least one of the function ran.

Some useful commands:

```shell
# only benchmarks
pytest -k benchmark
# no benchmarks
pytest -k "not benchmark"
# only linear layers benchmark, group by shape and if the input is contiguous or not 
pytest test/test_linear_layer.py --benchmark-group-by fullfunc,param:shape,param:contiguous
```

## Create new patterns to replace fx graph nodes

The first step to replace function/module calls in the graph is to create the pattern that will be replaced.
The easiest way to do this is to [convert the model to a fx graph](https://pytorch.org/docs/stable/fx.html), and then
print it with `utils.graph_report` or by printing the code `print(you_graph_module.code)`

Then you can use [replace_pattern](https://pytorch.org/docs/stable/fx.html#torch.fx.replace_pattern) to replace the
pattern in the graph. We have our own version of `replace_pattern` with some enhancements to work with modules for
example. You can find examples of that in `optimizer` folder.

## Code formatting

We use `black` / `isort` / `flake8` to format the code. You can run them with:

```shell
make source_code_format
make source_code_check_format
```
