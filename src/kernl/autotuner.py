from __future__ import annotations

import ast
import builtins
import inspect
import random
import re
import string
import textwrap
import threading
from collections import namedtuple
from typing import Dict

import torch
import triton
from torch._inductor.compile_fx import clone_preserve_strides
from triton.runtime.jit import KernelInterface, get_cuda_stream
from triton.testing import do_bench


class Autotuner(KernelInterface):
    def __init__(self, fn, arg_names, configs, key, signature, reset_to_zero, prune_configs_by: Dict = None):
        if not configs:
            self.configs = [Config(dict(), num_warps=4, num_stages=2)]
        else:
            self.configs = configs
        #fn_signature = inspect.signature(fn)
        #self.arg_names = [v.name for v in fn_signature.parameters.values()]
        self.arg_names = arg_names
        self.signature = signature
        self.key_idx = [self.arg_names.index(k) for k in key]
        self.cache = dict()
        self.save_cache_hook = None
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
        if prune_configs_by:
            perf_model, top_k = prune_configs_by["perf_model"], prune_configs_by["top_k"]
            if "early_config_prune" in prune_configs_by:
                early_config_prune = prune_configs_by["early_config_prune"]
        else:
            perf_model, top_k, early_config_prune = None, None, None
        self.perf_model, self.configs_top_k = perf_model, top_k
        self.early_config_prune = early_config_prune
        self.lock = threading.Lock()
        self.src = textwrap.dedent(inspect.getsource(fn))
        self.src = self.src[self.src.find("def") :]
        self.fn = fn
        cache_key = ''.join(random.sample(string.ascii_lowercase, 8))
        self.fn.cache_key = f"{self.fn.__name__}_{cache_key}"
        self.fn.src = self.src
        self.fn.parse = self.parse
        self.constexprs = [self.arg_names.index(ann) for ann in self.fn.__annotations__.keys()]

    def warmup(self, *args, **kwargs):
        self.nargs = dict(zip(self.arg_names, args))
        for config in self.prune_configs(kwargs):
            self.fn.warmup(
                *args,
                num_warps=config.num_warps,
                num_stages=config.num_stages,
                **kwargs,
                **config.kwargs,
            )
        self.nargs = None

    # we do not parse `src` in the constructor because
    # the user might want to monkey-patch self.src dynamically.
    # Our unit tests do this, for example.
    def parse(self):
        tree = ast.parse(self.src)
        assert isinstance(tree, ast.Module)
        assert len(tree.body) == 1
        assert isinstance(tree.body[0], ast.FunctionDef)
        return tree

    def bench(self, launcher, *args, grid, **kwargs):
        stream = get_cuda_stream(torch.cuda.current_device())

        def kernel_call():
            if launcher.config.pre_hook is not None:
                launcher.config.pre_hook({**zip(self.arg_names, args), **launcher.config.kwargs})
            launcher(
                *args,
                grid=grid,
                stream=stream,
                **kwargs
            )

        return do_bench(kernel_call)

    def precompile(self):
        with self.lock:
            if self.launchers:
                return
            self.launchers = [self._precompile_config(c) for c in self.configs]

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

        torch.cuda.set_device(torch.cuda.current_device())

        binary = triton.compile(
            self.fn,
            **compile_meta,
        )

        call_args = [arg for i, arg in enumerate(self.arg_names) if i not in self.constexprs]
        def_args = list(self.arg_names)
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
            def launcher({', '.join(def_args)}, grid, stream, **kwargs):
                #set_device(current_device())  # TODO(jansel): is this needed?
                if callable(grid):
                    grid = grid(grid_meta)
                grid_size = len(grid)
                grid_0 = grid[0]
                grid_1 = grid[1] if grid_size > 1 else 1
                grid_2 = grid[2] if grid_size > 2 else 1
                
                if stream is None and not warmup:
                    stream = get_cuda_stream(device)
                bin.c_wrapper(grid_0, grid_1, grid_2, bin.num_warps, bin.shared,
                              stream, bin.cu_function, {', '.join(call_args)})
            """.lstrip(),
            scope,
        )

        launcher = scope["launcher"]
        launcher.config = cfg
        return launcher

    def clone_preserve_strides(x):
        needed_size = (
                sum((shape - 1) * stride for shape, stride in zip(x.size(), x.stride())) + 1
        )
        buffer = torch.as_strided(x, (needed_size,), (1,)).clone()
        return torch.as_strided(buffer, x.size(), x.stride())

    def autotune_to_one_config(self, *args, **kwargs):
        """Do the actual autotuning"""
        # clone the input args to avoid autotune contaminating them if
        # the kernel does in-place stores
        cloned_args = [
            clone_preserve_strides(arg) if isinstance(arg, torch.Tensor) else arg
            for arg in args
        ]
        timings = {
            launcher: self.bench(launcher, *cloned_args, **kwargs)
            for launcher in self.launchers
        }
        self.launchers = [builtins.min(timings, key=timings.get)]
        if self.save_cache_hook:
            self.save_cache_hook(self.launchers[0].config)

    def run(self, *args, grid):
        stream = get_cuda_stream(torch.cuda.current_device())
        if len(self.launchers) != 1:
            if len(self.launchers) == 0:
                self.precompile()
            if len(self.launchers) > 1:
                self.autotune_to_one_config(*args, grid=grid)

        (launcher,) = self.launchers
        if launcher.config.pre_hook is not None:
            launcher.config.pre_hook(
                {**zip(self.arg_names, args), **launcher.config.kwargs}
            )
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
                )
            else:
                raise e

        return result

    def prune_configs(self, kwargs):
        pruned_configs = self.configs
        if self.early_config_prune:
            pruned_configs = self.early_config_prune(self.configs, self.nargs)
        if self.perf_model:
            top_k = self.configs_top_k
            if isinstance(top_k, float) and top_k <= 1.0:
                top_k = int(len(self.configs) * top_k)
            if len(pruned_configs) > top_k:
                est_timing = {
                    config: self.perf_model(
                        **self.nargs,
                        **kwargs,
                        **config.kwargs,
                        num_stages=config.num_stages,
                        num_warps=config.num_warps,
                    )
                    for config in pruned_configs
                }
                pruned_configs = sorted(est_timing.keys(), key=lambda x: est_timing[x])[:top_k]
        return pruned_configs


class Config:
    """
    An object that represents a possible kernel configuration for the auto-tuner to try.

    :ivar meta: a dictionary of meta-parameters to pass to the kernel as keyword arguments.
    :type meta: dict[Str, Any]
    :ivar num_warps: the number of warps to use for the kernel when compiled for GPUs. For example, if
                      `num_warps=8`, then each kernel instance will be automatically parallelized to
                      cooperatively execute using `8 * 32 = 256` threads.
    :type num_warps: int
    :ivar num_stages: the number of stages that the compiler should use when software-pipelining loops.
                       Mostly useful for matrix multiplication workloads on SM80+ GPUs.
    :type num_stages: int
    :ivar pre_hook: a function that will be called before the kernel is called. Parameters of this
                    function are args.
    """

    def __init__(self, kwargs, num_warps=4, num_stages=2, pre_hook=None):
        self.kwargs = kwargs
        self.num_warps = num_warps
        self.num_stages = num_stages
        self.pre_hook = pre_hook

    def __str__(self):
        res = []
        for k, v in self.kwargs.items():
            res.append(f"{k}: {v}")
        res.append(f"num_warps: {self.num_warps}")
        res.append(f"num_stages: {self.num_stages}")
        return ", ".join(res)


def autotune(configs, arg_names, key, signature, prune_configs_by=None, reset_to_zero=None):
    """
    Decorator for auto-tuning a :code:`triton.jit`'d function.

    .. highlight:: python
    .. code-block:: python

        @triton.autotune(configs=[
            triton.Config(meta={'BLOCK_SIZE': 128}, num_warps=4),
            triton.Config(meta={'BLOCK_SIZE': 1024}, num_warps=8),
          ],
          key=['x_size'] # the two above configs will be evaluated anytime
                         # the value of x_size changes
        )
        @triton.jit
        def kernel(x_ptr, x_size, **META):
            BLOCK_SIZE = META['BLOCK_SIZE']

    :note: When all the configurations are evaluated, the kernel will run multiple time.
           This means that whatever value the kernel updates will be updated multiple times.
           To avoid this undesired behavior, you can use the `reset_to_zero` argument, which
           reset the value of the provided tensor to `zero` before running any configuration.

    :param configs: a list of :code:`triton.Config` objects
    :type configs: list[triton.Config]
    :param key: a list of argument names whose change in value will trigger the evaluation of all provided configs.
    :type key: list[str]
    :param prune_configs_by: a dict of functions that are used to prune configs, fields:
        'perf_model': performance model used to predicate running time with different configs, returns running time
        'top_k': number of configs to bench
        'early_config_prune'(optional): a function used to do early prune (eg, num_stages). It take configs:List[Config] as its input, and returns pruned configs.
    :param reset_to_zero: a list of argument names whose value will be reset to zero before evaluating any configs.
    :type reset_to_zero: list[str]
    """

    def decorator(fn):
        return Autotuner(fn, arg_names, configs, key, signature, reset_to_zero, prune_configs_by)

    return decorator


class Heuristics(KernelInterface):
    def __init__(self, fn, arg_names, values) -> None:
        self.fn = fn
        self.values = values
        self.arg_names = arg_names

    def run(self, *args, **kwargs):
        for v, heur in self.values.items():
            kwargs[v] = heur({**dict(zip(self.arg_names, args)), **kwargs})
        return self.fn.run(*args, **kwargs)


def heuristics(values, arg_names):
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
        return Heuristics(fn, arg_names, values)

    return decorator
