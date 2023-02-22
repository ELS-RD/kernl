import ast
import builtins
import inspect
import logging
import re
import string
import random
import textwrap
import threading
from typing import Dict, List, Optional

import torch
import triton
from triton import Config, cdiv
from triton.runtime.jit import get_cuda_stream


log = logging.getLogger(__name__)


class KernelInterface:
    def __getitem__(self, grid):
        """
        A JIT function is launched with: fn[grid](*args, **kwargs).
        Hence JITFunction.__getitem__ returns a callable proxy that
        memorizes the grid.
        """
        stream = get_cuda_stream(torch.cuda.current_device())

        def launcher(*args, **kwargs):
            return self.run(*args, grid=grid, stream=stream, **kwargs)

        return launcher


class Autotuner(KernelInterface):
    """
    Simplified version of Triton autotuner.
    Unlike the main triton Autotuner, this version can precompile all
    configs, and does not rely on the Triton JIT.
    """

    def __init__(self, fn, configs, signature, key, reset_to_zero, prune_configs_by: Dict = None):
        super().__init__()
        self.launchers = []
        if not configs:
            self.configs = [Config(dict(), num_warps=4, num_stages=2)]
        else:
            self.configs = configs
        self.signature = signature
        fn_signature = inspect.signature(fn)
        self.arg_names = [v.name for v in fn_signature.parameters.values()]
        self.key_idx = [self.arg_names.index(k) for k in key]
        self.cache = dict()
        self.src = textwrap.dedent(inspect.getsource(fn))
        self.src = self.src[self.src.find("def") :]
        # hook to reset all required tensor to zeros before relaunching a kernel
        self.hook = lambda args: 0
        if reset_to_zero is not None:
            self.reset_idx = [self.arg_names.index(k) for k in reset_to_zero]

            def _hook(args):
                for i in self.reset_idx:
                    args[i].zero_()

            self.hook = _hook
        if prune_configs_by:
            perf_model, top_k = prune_configs_by["perf_model"], prune_configs_by["top_k"]
            if "early_config_prune" in prune_configs_by:
                early_config_prune = prune_configs_by["early_config_prune"]
        else:
            perf_model, top_k, early_config_prune = None, None, None
        self.perf_model, self.configs_top_k = perf_model, top_k
        self.early_config_prune = early_config_prune
        self.fn = fn
        self.fn.cache_key = ''.join(random.choice(string.printable) for i in range(20)) # TODO: fix the cache key
        self.__annotations__ = fn.__annotations__
        # index of constexprs
        self.constexprs = [self.arg_names.index(ann) for ann in self.__annotations__.keys()]
        self.fn.parse = self.parse
        self.fn.src = self.src
        self.lock = threading.Lock()

    def parse(self):
        tree = ast.parse(self.src)
        assert isinstance(tree, ast.Module)
        assert len(tree.body) == 1
        assert isinstance(tree.body[0], ast.FunctionDef)
        return tree

    def precompile(self, warm_cache_only_with_cc=None):
        with self.lock:
            if self.launchers:
                return
            self.launchers = [self._precompile_config(c, warm_cache_only_with_cc) for c in self.configs]
            self.configs = None

    @staticmethod
    def is_divisible_by_16(x):
        if hasattr(x, "data_ptr"):
            return x.data_ptr() % 16 == 0
        elif isinstance(x, int):
            return x % 16 == 0
        if x is None:
            return True
        return False

    def _precompile_config(self, cfg: Config, warm_cache_only_with_cc: int):
        """Ahead of time compile a given autotuner config."""
        # make constants:
        constexpr_args = [f"{arg}" for i, arg in enumerate(self.arg_names) if i in self.constexprs]
        constants = {i: k for i, k in zip(self.constexprs, constexpr_args)}
        for k, v in constants.items():
            constants[k] = cfg.kwargs[v] if v in cfg.kwargs.keys() else 1
        compile_meta = {"constants": constants, "signature": self.signature, "num_warps": cfg.num_warps, "num_stages": cfg.num_stages}
        cfg.divisible_by_16 = [i for i, arg in enumerate(self.arg_names) if self.is_divisible_by_16(arg)]
        cfg.equal_to_1 = [i for i, arg in enumerate(self.arg_names) if isinstance(arg, int) and arg == 1]

        if warm_cache_only_with_cc:
            triton.compile(
                self.fn,
                warm_cache_only=True,
                cc=warm_cache_only_with_cc,
                **compile_meta,
            )
            return

        torch.cuda.set_device(torch.cuda.current_device())
        compile_meta["device"] = torch.cuda.current_device()

        binary = triton.compile(
            self.fn,
            configs=[cfg],
            **compile_meta,
        )

        call_args = [arg for i, arg in enumerate(self.arg_names) if i not in self.constexprs and arg != "stream"]
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
            def launcher({', '.join(def_args)}, grid, stream):
                # set_device(current_device())  # TODO(jansel): is this needed?
                if callable(grid):
                    grid = grid(grid_meta)
                grid_size = len(grid)
                grid_0 = grid[0]
                grid_1 = grid[1] if grid_size > 1 else 1
                grid_2 = grid[2] if grid_size > 2 else 1
                bin.c_wrapper(grid_0, grid_1, grid_2, bin.num_warps, bin.shared, stream, bin.cu_function, None, None, None, {', '.join(call_args)})
            """.lstrip(),
            scope,
        )

        launcher = scope["launcher"]
        launcher.config = cfg
        return launcher

    def bench(self, launcher, *args, grid, **kwargs):
        """Measure the performance of a given launcher"""

        current = dict(**kwargs, **launcher.config.kwargs)

        def kernel_call():
            if launcher.config.pre_hook is not None:
                launcher.config.pre_hook({**zip(self.arg_names, args), **launcher.config.kwargs})
            launcher(*args, grid=grid, **current)

        from triton.testing import do_bench

        return do_bench(kernel_call)

    @staticmethod
    def clone_preserve_strides(x):
        needed_size = sum((shape - 1) * stride for shape, stride in zip(x.size(), x.stride())) + 1
        buffer = torch.as_strided(x, (needed_size,), (1,)).clone()
        return torch.as_strided(buffer, x.size(), x.stride())

    def autotune_to_one_config(self, *args, **kwargs):
        """Do the actual autotuning"""

        # clone the input args to avoid autotune contaminating them if
        # the kernel does in-place stores
        cloned_args = [self.clone_preserve_strides(arg) if isinstance(arg, torch.Tensor) else arg for arg in args]
        timings = {launcher: self.bench(launcher, *cloned_args, **kwargs) for launcher in self.launchers}
        self.launchers = [builtins.min(timings, key=timings.get)]

    def run(self, *args, grid, **kwargs):
        stream = get_cuda_stream(torch.cuda.current_device())
        if len(self.launchers) != 1:
            if len(self.launchers) == 0:
                self.precompile()
            if len(self.launchers) > 1:
                self.autotune_to_one_config(*args, grid=grid, **kwargs)

        (launcher,) = self.launchers
        if launcher.config.pre_hook is not None:
            launcher.config.pre_hook({**zip(self.arg_names, args), **launcher.config.kwargs})
        try:
            result = launcher(*args, grid=grid, stream=stream, **kwargs)
        except TypeError as e:
            if re.match(r"function takes exactly \d+ arguments \(\d+ given\)", str(e)):
                raise RuntimeError(
                    """Consider updating Triton with
`pip install -U "git+https://github.com/openai/triton@af76c989eb4799b015f8b288ccd8421558772e56#subdirectory=python"`"""
                )
            else:
                raise e

        return result


def autotune(
    configs: List[Config],
    key: List[str],
    signature: Dict[int, str],
    reset_to_zero: Optional[List[str]] = None,
    prune_configs_by: Optional[Dict] = None,
):
    """
    A copy of triton.autotune that calls our subclass.
    """
    configs = unique_configs(configs)

    def decorator(fn):
        return Autotuner(
            fn,
            configs=configs,
            signature=signature,
            key=key,
            reset_to_zero=reset_to_zero,
            prune_configs_by=prune_configs_by,
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


def grid(xnumel, ynumel=None, znumel=None):
    """Helper function to compute triton grids"""

    if ynumel and znumel:

        def grid_fn(meta):
            return (
                cdiv(xnumel, meta["BLOCK_M"]),
                cdiv(ynumel, meta["BLOCK_N"]),
                cdiv(znumel, meta["BLOCK_K"]),
            )

    elif ynumel:

        def grid_fn(meta):
            return (
                cdiv(xnumel, meta["BLOCK_M"]),
                cdiv(ynumel, meta["BLOCK_N"]),
                1,
            )

    else:

        def grid_fn(meta):
            return (
                cdiv(xnumel, meta["BLOCK_M"]),
                1,
                1,
            )

    return grid_fn
