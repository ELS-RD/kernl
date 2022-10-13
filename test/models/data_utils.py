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

from typing import Callable, Dict

import torch
from transformers import BertPreTrainedModel, T5PreTrainedModel


def get_attention_mask(shape: (int, int)) -> torch.Tensor:
    bs, seq_len = shape
    return (
        torch.randint(1, seq_len, (bs,), device="cuda")[:, None] > torch.arange(0, seq_len, device="cuda")[None, :]
    ).to(torch.int64)


def get_causal_mask(shape: (int, int)) -> torch.Tensor:
    batch, seq_length = shape
    return torch.tril(torch.ones((batch, seq_length, seq_length), dtype=torch.int64, device="cuda"))


def get_input(model: Callable, shape: (int, int), is_causal: bool = False) -> Dict[str, torch.Tensor]:
    result = {
        "input_ids": torch.randint(2, 1000, size=shape, dtype=torch.int64, device="cuda"),
    }

    if is_causal:
        result["attention_mask"] = get_causal_mask(shape)
    else:
        result["attention_mask"] = get_attention_mask(shape)

    if isinstance(model, BertPreTrainedModel):
        result["token_type_ids"] = torch.ones(size=shape, dtype=torch.int64, device="cuda")

    if isinstance(model, T5PreTrainedModel):
        result["decoder_input_ids"] = torch.randint(2, 1000, size=shape, dtype=torch.int64, device="cuda")

    return result
