import pytest
import torch

from kernl.autocast import autocast, custom_fwd, get_autocast_dtype, is_autocast_enabled, is_torch_autocast_dtype_used


def test_autocast():

    with autocast():
        assert is_autocast_enabled() is True

    with autocast(enabled=False):
        assert is_autocast_enabled() is False
    assert is_autocast_enabled() is True

    with autocast(dtype=torch.float32):
        assert get_autocast_dtype() is torch.float32
    assert get_autocast_dtype() is torch.float16

    with autocast(use_torch_autocast_dtype=False):
        assert is_torch_autocast_dtype_used() is False
    assert is_torch_autocast_dtype_used() is True


@pytest.mark.parametrize(
    """
    torch_autocast_enabled, kernl_autocast_enabled, torch_autocast_dtype,
    kernl_autocast_dtype, use_torch_autocast_dtype, cast_inputs, expected_dtype""",
    [
        # we first check if @custom_fwd behaves the same as torch.cuda.amp.custom_fwd
        # when we're not using kernl autocast
        (False, False, None, None, False, None, torch.float32),
        (False, False, None, None, False, torch.bfloat16, torch.float32),
        (True, False, torch.float16, None, False, None, torch.float32),
        (True, False, torch.float16, None, False, torch.bfloat16, torch.bfloat16),
        # we check @custom_fwd without torch autocast
        (False, True, None, torch.float16, False, None, torch.float16),
        (False, True, None, torch.float16, False, torch.bfloat16, torch.bfloat16),
        (False, True, None, torch.float16, True, None, torch.float32),
        (False, True, None, torch.float16, True, torch.bfloat16, torch.bfloat16),
        # we check @custom_fwd within torch autocast context and kernl autocast context
        (True, True, torch.float16, torch.bfloat16, False, None, torch.bfloat16),
        (True, True, torch.float16, torch.float16, False, torch.bfloat16, torch.bfloat16),
        (True, True, torch.float16, torch.bfloat16, False, None, torch.bfloat16),
        (True, True, torch.float16, torch.bfloat16, True, None, torch.float16),
        (True, True, torch.float16, torch.float16, True, torch.bfloat16, torch.bfloat16),
    ],
)
def test_custom_fwd(
    torch_autocast_enabled,
    kernl_autocast_enabled,
    torch_autocast_dtype,
    kernl_autocast_dtype,
    use_torch_autocast_dtype,
    cast_inputs,
    expected_dtype,
):
    @custom_fwd(cast_inputs=cast_inputs)
    def fwd(inputs: torch.Tensor):
        assert inputs.dtype == expected_dtype

    with torch.cuda.amp.autocast(enabled=torch_autocast_enabled, dtype=torch_autocast_dtype), autocast(
        enabled=kernl_autocast_enabled, dtype=kernl_autocast_dtype, use_torch_autocast_dtype=use_torch_autocast_dtype
    ):
        inputs = torch.empty((100), dtype=torch.float32, device="cuda")
        fwd(inputs)
        print
