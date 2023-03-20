import torch
import functools

from typing import Any

from kernl.utils.cast import _cast


_is_autocast_enabled: bool = True
_autocast_dtype: torch.dtype = torch.float16
_is_torch_autocast_dtype_used: bool = True


def enable_autocast(is_autocast_enabled: bool):
    """Change autocast state for kernl.
    Args:
        is_autocast_enabled: True to enable autocast, False to disable
    """
    global _is_autocast_enabled
    _is_autocast_enabled = is_autocast_enabled


def is_autocast_enabled() -> bool:
    """Check is autocast enabled.
    Returns:
        wether autocast is enabled
    """
    return _is_autocast_enabled


def set_autocast_dtype(dtype: torch.dtype) -> None:
    """Change autocast target dtype.
    Args:
        dtype: which dtype to autocast on
    """
    global _autocast_dtype
    _autocast_dtype = dtype


def get_autocast_dtype() -> torch.dtype:
    """Get autocast target dtype.
    Returns:
        autocast target dype
    """
    return _autocast_dtype


def use_torch_autocast_dtype(use_torch_autocast_dtype: bool) -> None:
    """ Check i
    Args:
        use_torch_autocast_dtype: wether torch autocast dtype is used
    """
    global _is_torch_autocast_dtype_used
    _is_torch_autocast_dtype_used = use_torch_autocast_dtype


def is_torch_autocast_dtype_used() -> bool:
    """ Check if torch autocast dtype is used in autocast
    Returns:
        wether torch autocast dtype is used
    """
    return _is_torch_autocast_dtype_used


class autocast(object):
    """Serve as context managers or decorators that allow fused kernels to run in mixed precision.

    In these regions, kernels run in an op-specific dtype chosen by autocast
    to improve performance while maintaining accuracy.

    When entering an autocast-enabled region, Tensors may be any type.
    You should not call `half()` or `bfloat16()` on your model(s) or inputs when using autocasting.

    `autocast` should wrap only the forward pass(es) of your network, including the loss
    computation(s).  Backward passes under autocast are not recommended.
    Backward ops run in the same type that autocast used for corresponding forward ops.

    Heavily inspired by [torch.autocast][].
    Args:
        enabled:  Whether autocasting should be enabled in the region.
        dtype:  Whether to use torch.float16 or torch.bfloat16.
        use_torch_autocast_dtype:  Whether use torch autocast dtype.
    """
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
        inputs = cast_inputs
        if inputs is None:
            if is_autocast_enabled():
                if is_torch_autocast_dtype_used():
                    if torch.is_autocast_enabled():
                        inputs = torch.get_autocast_gpu_dtype()
                else:
                    inputs = get_autocast_dtype()
        if inputs is None:
            args[0]._fwd_used_autocast = torch.is_autocast_enabled()
            return fwd(*args, **kwargs)
        else:
            autocast_context = torch.is_autocast_enabled() or is_autocast_enabled()
            args[0]._fwd_used_autocast = torch.is_autocast_enabled()
            if autocast_context:
                with torch.cuda.amp.autocast(enabled=False):
                    return fwd(*_cast(args, inputs), **_cast(kwargs, inputs))
            else:
                return fwd(*args, **kwargs)

    return decorate_fwd
