import builtins
import copy
import hashlib
import json
import logging
import os.path
import re
import threading
from typing import List, Dict, Optional

import torch

import triton
from triton import cdiv, Config
from triton.runtime.jit import get_cuda_stream, KernelInterface

from utils.autotuner_helper import type_of, key_of

log = logging.getLogger(__name__)

class Autotuner(KernelInterface):
    """
    Simplified version of Triton autotuner.
    Unlike the main triton Autotuner, this version can precompile all
    configs, and does not rely on the Triton JIT.
    """

    def __init__(self, fn, configs, key, reset_to_zero, prune_configs_by: Dict = None):
        super().__init__()
        self.launchers = []
        if not configs:
            self.configs = [Config(dict(), num_warps=4, num_stages=2)]
        else:
            self.configs = configs
        self.arg_names = fn.arg_names
        self.key_idx = [self.arg_names.index(k) for k in key]
        self.cache = dict()
        # hook to reset all required tensor to zeros before relaunching a kernel
        self.hook = lambda args: 0
        if reset_to_zero is not None:
            self.reset_idx = [self.arg_names.index(k) for k in reset_to_zero]

            def _hook(args):
                for i in self.reset_idx:
                    args[i].zero_()
            self.hook = _hook
        if prune_configs_by:
            perf_model, top_k = prune_configs_by['perf_model'], prune_configs_by['top_k']
            if 'early_config_prune' in prune_configs_by:
                early_config_prune = prune_configs_by['early_config_prune']
        else:
            perf_model, top_k, early_config_prune = None, None, None
        self.perf_model, self.configs_top_k = perf_model, top_k
        self.early_config_prune = early_config_prune
        self.fn = fn
        self.__annotations__ = fn.__annotations__
        # index of constexprs
        self.constexprs = [self.arg_names.index(ann) for ann in self.__annotations__.keys()]
        self.lock = threading.Lock()

    def precompile(self, warm_cache_only_with_cc=None):
        breakpoint()
        with self.lock:
            if self.launchers:
                return
            self.launchers = [
                self._precompile_config(c, warm_cache_only_with_cc)
                for c in self.configs
            ]
            self.configs = None

    def _precompile_config(self, cfg: Config, warm_cache_only_with_cc: int):
        """Ahead of time compile a given autotuner config."""
        all_args = {', '.join([f'{arg}' for arg in self.arg_names])},
        signature = {{i: type_of(key_of(arg)) for i, arg in enumerate(all_args) if i not in self.constexprs}}
        compile_meta = {
            "constants": dict(), "signature": signature
        }
        for k, v in cfg.kwargs.items():
            compile_meta["constants"][self.fn.arg_names.index(k)] = v
        compile_meta["num_warps"] = cfg.num_warps
        compile_meta["num_stages"] = cfg.num_stages

        if warm_cache_only_with_cc:
            triton.compile(
                self.fn,
                warm_cache_only=True,
                cc=warm_cache_only_with_cc,
                **compile_meta,
            )
            return

        torch.cuda.set_device(torch.cuda.current_device())

        binary = triton.compile(
            self.fn,
            **compile_meta,
        )

        call_args = [
            arg
            for i, arg in enumerate(self.fn.arg_names)
            if i not in self.fn.constexprs
        ]
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
                # set_device(current_device())  # TODO(jansel): is this needed?
                grid_0, grid_1, grid_2 = grid(grid_meta)
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
                launcher.config.pre_hook(
                    {**zip(self.arg_names, args), **launcher.config.kwargs}
                )
            launcher(
                *args,
                grid=grid,
                stream=stream,
            )

        from triton.testing import do_bench

        return do_bench(kernel_call)

    @staticmethod
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
            self.clone_preserve_strides(arg) if isinstance(arg, torch.Tensor) else arg
            for arg in args
        ]
        timings = {
            launcher: self.bench(launcher, *cloned_args, **kwargs)
            for launcher in self.launchers
        }
        self.launchers = [builtins.min(timings, key=timings.get)]

    def run(self, *args, grid, stream):
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


def hash_configs(configs: List[Config]):
    """
    Hash used to check for changes in configurations
    """
    hasher = hashlib.sha256()
    for cfg in configs:
        hasher.update(
            f"{sorted(cfg.kwargs.items())} {cfg.num_warps} {cfg.num_stages}\n".encode(
                "utf-8"
            )
        )
    return hasher.hexdigest()


def load_cached_autotuning(
    cache_filename: str, configs_hash: str, configs: List[Config]
):
    """
    Read a cached autotuning result from disk
    """
    if not os.path.exists(cache_filename):
        return None

    best_config = json.loads(open(cache_filename).read())
    if best_config.get("configs_hash") != configs_hash:
        return None

    matching_configs = [
        cfg
        for cfg in configs
        if all(val == best_config.get(key) for key, val in cfg.kwargs.items())
    ]
    if len(matching_configs) != 1:
        return None

    return matching_configs[0]


def autotune(
    configs: List[Config],
    key: List[str],
    reset_to_zero: Optional[List[str]] = None,
    prune_configs_by: Optional[Dict] = None,
):
    """
    A copy of triton.autotune that calls our subclass.
    """
    configs = unique_configs(configs)

    def decorator(fn):
        return Autotuner(
            fn, configs=configs, key=key, reset_to_zero=reset_to_zero, prune_configs_by=prune_configs_by,
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
                cdiv(xnumel, meta["XBLOCK"]),
                cdiv(ynumel, meta["YBLOCK"]),
                cdiv(znumel, meta["ZBLOCK"]),
            )

    elif ynumel:

        def grid_fn(meta):
            return (
                cdiv(xnumel, meta["XBLOCK"]),
                cdiv(ynumel, meta["YBLOCK"]),
                1,
            )

    else:

        def grid_fn(meta):
            return (
                cdiv(xnumel, meta["XBLOCK"]),
                1,
                1,
            )

    return grid_fn
