import ast
import builtins
import copy
import inspect
import logging
import re
import textwrap
import threading
from collections import namedtuple
from typing import Dict, List, Optional

import torch
import triton
from triton import Config, cdiv
from triton.runtime.jit import KernelInterface, get_cuda_stream
from triton.testing import do_bench


log = logging.getLogger(__name__)


class KernlAutotuner(KernelInterface):

    """
    Simplified version of Triton autotuner.
    Unlike the main triton Autotuner, this version can precompile all
    configs, and does not rely on the Triton JIT.
    """

    def __init__(self, fn, configs, key, reset_to_zero, prune_configs_by: Dict = None):
        if not configs:
            self.configs = [Config(dict(), num_warps=4, num_stages=2)]
        else:
            self.configs = configs
        self.fn = fn
        self.src = textwrap.dedent(inspect.getsource(fn))
        self.src = self.src[self.src.find("def") :]
        self.signature = inspect.signature(fn)
        self.arg_names = [v.name for v in self.signature.parameters.values()]
        self.key_idx = [self.arg_names.index(k) for k in key]
        self.launchers = list()
        # hook to reset all required tensor to zeros before relaunching a kernel
        self.hook = lambda args: 0
        if reset_to_zero is not None:
            self.reset_idx = [self.arg_names.index(k) for k in reset_to_zero]

            def _hook(args):
                for i in self.reset_idx:
                    args[i].zero_()

            self.hook = _hook
        # prune configs
        early_config_prune = None
        if prune_configs_by:
            perf_model, top_k = prune_configs_by["perf_model"], prune_configs_by["top_k"]
            if "early_config_prune" in prune_configs_by:
                early_config_prune = prune_configs_by["early_config_prune"]
        else:
            perf_model, top_k = None, None
        self.perf_model, self.configs_top_k = perf_model, top_k
        self.early_config_prune = early_config_prune
        self.constexprs = [self.arg_names.index(ann) for ann in self.fn.__annotations__.keys()]
        self.fn.parse = self.parse
        self.lock = threading.Lock()

    def parse(self):
        tree = ast.parse(self.src)
        assert isinstance(tree, ast.Module)
        assert len(tree.body) == 1
        assert isinstance(tree.body[0], ast.FunctionDef)
        return tree

    def precompile(self):
        with self.lock:
            if self.launchers:
                return
            self.launchers = [self._precompile_config(c) for c in self.configs]
            self.configs = None

    def _precompile_config(self, cfg: Config):
        """Ahead of time compile a given autotuner config."""
        compile_meta = dict()
        compile_meta["constants"] = dict()
        config = namedtuple("instance_descriptor", ["divisible_by_16", "equal_to_1"])(tuple(range(4)), ())
        compile_meta["configs"] = [config]
        for k, v in cfg.kwargs.items():
            compile_meta["constants"][self.arg_names.index(k)] = v
        compile_meta["num_warps"] = cfg.num_warps
        compile_meta["num_stages"] = cfg.num_stages
        compile_meta["device"] = torch.cuda.current_device()
        compile_meta["signature"] = self.signature

        # load binary to the correct device
        with torch.cuda.device(compile_meta["device"]):
            # need to initialize context
            torch.cuda.synchronize(torch.cuda.current_device())
            binary = triton.compile(
                self.fn,
                **compile_meta,
            )

        call_args = [arg for i, arg in enumerate(self.fn.arg_names) if i not in self.fn.constexprs]
        def_args = list(self.fn.arg_names)
        while def_args and def_args[-1] in cfg.kwargs:
            def_args.pop()

        scope = {
            "grid_meta": cfg.kwargs,
            "bin": binary,
            "torch": torch,
            "set_device": torch.cuda.set_device,
            "current_device": torch.cuda.current_device,
        }
        exec(
            f"""
            def launcher({', '.join(def_args)}, grid, stream):
                if callable(grid):
                    grid_0, grid_1, grid_2 = grid(grid_meta)
                else:
                    grid_0, grid_1, grid_2 = grid
                bin.c_wrapper(grid_0, grid_1, grid_2, bin.num_warps, bin.shared,
                            stream, bin.cu_function, None, None, None,
                            {', '.join(call_args)})
            """.lstrip(),
            scope,
        )

        launcher = scope["launcher"]
        launcher.config = cfg
        return launcher

    def bench(self, launcher, *args, grid):
        """Measure the performance of a given launcher"""
        stream = get_cuda_stream(torch.cuda.current_device())

        def kernel_call():
            if launcher.config.pre_hook is not None:
                launcher.config.pre_hook({**zip(self.arg_names, args), **launcher.config.kwargs})
            launcher(
                *args,
                grid=grid,
                stream=stream,
            )

        return do_bench(kernel_call, rep=40, fast_flush=True)

    def autotune_to_one_config(self, *args, **kwargs):
        """Do the actual autotuning"""

        # clone inplace buffers to avoid autotune contaminating them if
        # the kernel does in-place stores. avoid cloning other buffers because
        # it leads to increase memory use
        cloned_args = []
        for i, arg in enumerate(args):
            if self.fn.arg_names[i] in self.mutated_arg_names:
                assert isinstance(arg, torch.Tensor)
                cloned_args.append(clone_preserve_strides(arg))
            else:
                cloned_args.append(arg)

        timings = {launcher: self.bench(launcher, *cloned_args, **kwargs) for launcher in self.launchers}
        self.launchers = [builtins.min(timings, key=timings.get)]

    def run(self, *args, grid, stream):
        if len(self.launchers) != 1:
            if len(self.launchers) == 0:
                self.precompile()
            if len(self.launchers) > 1:
                self.autotune_to_one_config(*args, grid=grid)

        (launcher,) = self.launchers
        if launcher.config.pre_hook is not None:
            launcher.config.pre_hook({**zip(self.arg_names, args), **launcher.config.kwargs})
        try:
            result = launcher(
                *args,
                grid=grid,
                stream=stream,
            )
        except TypeError as e:
            if re.match(r"function takes exactly \d+ arguments \(\d+ given\)", str(e)):
                raise RuntimeError(
                    """Consider updating Triton with
`pip install -U "git+https://github.com/openai/triton@af76c989eb4799b015f8b288ccd8421558772e56#subdirectory=python"`"""
                ) from e
            else:
                raise e

        return result


def kernl_autotune(
    configs: List[Config],
    key: List[str],
    reset_to_zero: Optional[List[str]] = None,
    prune_configs_by: Optional[Dict] = None,
):
    """
    A copy of triton.autotune that calls our subclass.  Our subclass
    has additional debugging, error handling, and on-disk caching.
    """
    configs = unique_configs(configs)

    def decorator(fn):
        return KernlAutotuner(
            fn, configs=configs, key=key, reset_to_zero=reset_to_zero, prune_configs_by=prune_configs_by
        )

    return decorator


def unique_configs(configs: List[Config]):
    """Remove duplicate configurations"""
    seen = set()
    pruned_configs = []
    for cfg in configs:
        key = tuple(cfg.kwargs.items())
        if key not in seen:
            seen.add(key)
            pruned_configs.append(cfg)
    return pruned_configs


def clone_preserve_strides(x):
    needed_size = sum((shape - 1) * stride for shape, stride in zip(x.size(), x.stride())) + 1
    buffer = torch.as_strided(x, (needed_size,), (1,)).clone()
    return torch.as_strided(buffer, x.size(), x.stride())


def grid(xnumel, ynumel=None, znumel=None):
    """Helper function to compute triton grids"""

    def get_grid_dim(numel, block_name, block):
        if numel is None:
            return 1
        label = block_name[0]
        if numel == 1:
            assert block == 1, (
                f"TritonKernel.indexing assumes {label.lower()}numel == 1 => {block_name} == 1"
                f"({label.lower()}numel=={numel}, {block_name}={block})."
            )
        return cdiv(numel, block)

    def grid_fn(meta):
        return (
            get_grid_dim(xnumel, "XBLOCK", meta.get("XBLOCK", None)),
            get_grid_dim(ynumel, "YBLOCK", meta.get("YBLOCK", None)),
            get_grid_dim(znumel, "ZBLOCK", meta.get("ZBLOCK", None)),
        )

    return grid_fn


class KernlHeuristics(KernelInterface):
    def __init__(self, fn, values) -> None:
        self.fn = fn
        self.values = values
        signature = inspect.signature(fn)
        self.arg_names = [v.name for v in signature.parameters.values()]

    def run(self, *args, **kwargs):
        for v, heur in self.values.items():
            kwargs[v] = heur({**dict(zip(self.arg_names, args)), **kwargs})
        return self.fn.run(*args, **kwargs)


def kernl_heuristics(values):
    """
    Decorator for specifying how the values of certain meta-parameters may be computed.
    This is useful for cases where auto-tuning is prohibitevely expensive, or just not applicable.
    .. highlight:: python
    .. code-block:: python
        @triton.heuristics(values={'BLOCK_SIZE': lambda args: 2 ** int(math.ceil(math.log2(args[1])))})
        @triton.jit
        def kernel(x_ptr, x_size, **META):
            BLOCK_SIZE = META['BLOCK_SIZE'] # smallest power-of-two >= x_size
    .param values: a dictionary of meta-parameter names and functions that compute the value of the meta-parameter.
                   each such function takes a list of positional arguments as input.
    .type values: dict[str, Callable[[list[Any]], Any]]
    """

    def decorator(fn):
        return KernlHeuristics(fn, values)

    return decorator
