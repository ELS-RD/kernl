#  Copyright 2022 Lefebvre Sarrut
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

import torch


def check_all_close(a: torch.Tensor, b: torch.Tensor, rtol=0, atol=1e-1) -> None:
    """
    Check that all elements of a and b are close.
    """
    assert a.shape == b.shape, f"Shapes don't match: {a.shape} != {b.shape}"
    assert a.dtype == b.dtype, f"Dtypes don't match: {a.dtype} != {b.dtype}"
    assert a.device == b.device, f"Devices don't match: {a.device} != {b.device}"
    max_abs_diff = torch.max(torch.abs(a - b))
    max_rel_diff = torch.max(torch.abs(a / b))
    mismatch_elements = torch.sum(torch.abs(a - b) > atol + rtol * torch.abs(b))
    nb_elements = torch.numel(a)
    msg = (
        f"Differences: "
        f"{max_abs_diff} (max abs), "
        f"{max_rel_diff} (max rel), "
        f"{mismatch_elements}/{nb_elements} (mismatch elements)"
    )
    assert torch.allclose(a, b, rtol=rtol, atol=atol), msg
