import builtins
import copy
import logging
import re
import threading
from typing import List

import torch
import triton
from triton import cdiv, Config
from triton.runtime.jit import get_cuda_stream, KernelInterface
from triton.testing import do_bench

log = logging.getLogger(__name__)

class KernlAutotuner(KernelInterface):

    """
    Simplified version of Triton autotuner.
    Unlike the main triton Autotuner, this version can precompile all
    configs, and does not rely on the Triton JIT.
    """

    def __init__(self, fn, meta, configs, mutated_arg_names):
        super().__init__()
        self.fn = fn
        self.meta = meta
        self.mutated_arg_names = mutated_arg_names
        self.configs = configs
        self.launchers = []
        self.lock = threading.Lock()


    def precompile(self, warm_cache_only_with_cc=None):
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
        compile_meta = copy.deepcopy(self.meta)
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

        # load binary to the correct device
        with torch.cuda.device(compile_meta["device"]):
            # need to initialize context
            torch.cuda.synchronize(torch.cuda.current_device())
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
                launcher.config.pre_hook(
                    {**zip(self.arg_names, args), **launcher.config.kwargs}
                )
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
                ) from e
            else:
                raise e

        return result

def kernl_autotune(
    configs: List[Config],
    meta,
):
    """
    A copy of triton.autotune that calls our subclass.  Our subclass
    has additional debugging, error handling, and on-disk caching.
    """
    configs = unique_configs(configs)
    assert len(configs) == 1

    mutated_arg_names = meta.pop("mutated_arg_names", ())
    def decorator(fn):
        return KernlAutotuner(
            fn,
            meta=meta,
            configs=configs,
            mutated_arg_names=mutated_arg_names,
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

def template(num_stages, num_warps, meta):
    """
    Compile a triton template
    """
    return kernl_autotune(
        [triton.Config({}, num_stages=num_stages, num_warps=num_warps)], meta=meta
    )


def clone_preserve_strides(x):
    needed_size = (
        sum((shape - 1) * stride for shape, stride in zip(x.size(), x.stride())) + 1
    )
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
