import collections
import functools

import torch
from torch._six import string_classes


try:
    import numpy as np

    HAS_NUMPY = True
except ModuleNotFoundError:
    np = None  # type: ignore[assignment]
from typing import Any


_is_autocast_enabled: bool = False
_autocast_dtype: torch.dtype = torch.float16
_is_torch_autocast_dtype_used: bool = False


def is_autocast_enabled() -> bool:
    return _is_autocast_enabled


def enable_autocast(is_autocast_enabled: bool):
    global _is_autocast_enabled
    _is_autocast_enabled = is_autocast_enabled


def set_autocast_dtype(dtype: torch.dtype) -> None:
    global _autocast_dtype
    _autocast_dtype = dtype


def get_autocast_dtype() -> torch.dtype:
    return _autocast_dtype


def use_torch_autocast_dtype(use_torch_autocast_dtype: bool) -> None:
    global _is_torch_autocast_dtype_used
    _is_torch_autocast_dtype_used = use_torch_autocast_dtype


def is_torch_autocast_dtype_used() -> bool:
    return _is_torch_autocast_dtype_used


class autocast(object):
    def __init__(
        self, enabled: bool = True, dtype: torch.dtype = torch.float16, use_torch_autocast_dtype: bool = False
    ):
        self._enabled = enabled
        self._dtype = dtype
        self._use_autocast_dtype = use_torch_autocast_dtype

    def __enter__(self):
        self._prev = is_autocast_enabled()
        self._prev_dtype = get_autocast_dtype()
        self._prev_use_torch_autocast_dtype = is_torch_autocast_dtype_used()
        enable_autocast(self._enabled)
        set_autocast_dtype(self._dtype)
        use_torch_autocast_dtype(self._use_autocast_dtype)

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any):
        enable_autocast(self._prev)
        set_autocast_dtype(self._prev_dtype)
        use_torch_autocast_dtype(self._prev_use_torch_autocast_dtype)


def custom_fwd(fwd=None, **kwargs):
    """
    Helper decorator for ``forward`` methods of custom autograd functions (subclasses of
    :class:`torch.autograd.Function`).  See the :ref:`example page<amp-custom-examples>` for more detail.

    Args:
        cast_inputs (:class:`torch.dtype` or None, optional, default=None):  If not ``None``,
            when ``forward`` runs in an autocast-enabled region, casts incoming
            floating-point CUDA Tensors to the target dtype (non-floating-point Tensors are not affected),
            then executes ``forward`` with autocast disabled.
            If ``None``, ``forward``'s internal ops execute with the current autocast state.

    .. note::
        If the decorated ``forward`` is called outside an autocast-enabled region,
        :func:`custom_fwd<custom_fwd>` is a no-op and ``cast_inputs`` has no effect.
    """
    if fwd is None:
        if len(kwargs) == 0:
            cast_inputs = None
        else:
            assert len(kwargs) == 1
            cast_inputs = kwargs["cast_inputs"]

        return functools.partial(custom_fwd, cast_inputs=cast_inputs)

    if len(kwargs) == 0:
        cast_inputs = None
    else:
        assert len(kwargs) == 1
        cast_inputs = kwargs["cast_inputs"]

    @functools.wraps(fwd)
    def decorate_fwd(*args, **kwargs):

        print(f"cast_inputs: {cast_inputs}")
        inputs = cast_inputs
        if inputs is None:
            if is_autocast_enabled():
                if is_torch_autocast_dtype_used():
                    if torch.is_autocast_enabled():
                        inputs = torch.get_autocast_gpu_dtype()
                else:
                    inputs = get_autocast_dtype()
            else:
                if torch.is_autocast_enabled():
                    inputs = torch.get_autocast_gpu_dtype()
        print(f"cast_inputs 2: {inputs}")
        args[0]._fwd_used_autocast = torch.is_autocast_enabled()
        if inputs is None:
            return fwd(*args, **kwargs)
        else:
            autocast_context = torch.is_autocast_enabled()
            if autocast_context:
                with torch.cuda.amp.autocast(enabled=False):
                    return fwd(*_cast(args, inputs), **_cast(kwargs, inputs))
            if inputs is None:
                return fwd(*args, **kwargs)

    return decorate_fwd


# Casts Tensors and containers of Tensors.  Special-cases passthroughs for strings and np.ndarrays, which
# may be falsely detected as "Iterables."
def _cast(value, dtype):
    print(f"to {dtype}")
    if isinstance(value, torch.Tensor):
        is_eligible = value.is_floating_point() and value.is_cuda and (value.dtype is not torch.float64)
        return value.to(dtype) if is_eligible else value
    elif isinstance(value, string_classes):
        return value
    elif HAS_NUMPY and isinstance(value, np.ndarray):
        return value
    elif isinstance(value, collections.abc.Mapping):
        return {_cast(k, dtype): _cast(v, dtype) for k, v in value.items()}
    elif isinstance(value, collections.abc.Iterable):
        iterable = map(lambda v: _cast(v, dtype), value)
        if isinstance(value, list) or isinstance(value, tuple):
            return type(value)(iterable)
        else:
            return iterable
    else:
        return value
