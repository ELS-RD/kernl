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

from typing import Dict

from typing import Dict, Callable
from transformers import BertPreTrainedModel, T5PreTrainedModel
import torch


def get_attention_mask(shape: (int, int)) -> torch.Tensor:
    return torch.randint(1, shape[1], (shape[0],), device="cuda")[:, None] > torch.arange(0, shape[1], device="cuda")[
                                                                             None, :]

def get_input_causal(model: Callable, shape: (int, int)) -> Dict[str, torch.Tensor]:
    batch, seq_length = shape
    mask = torch.tril(torch.ones((batch, seq_length, seq_length), dtype=torch.int64, device="cuda"))
    result = get_input_non_causal(model, shape)
    result['attention_mask'] = mask
    return result


def get_input_non_causal(model: Callable, shape: (int, int)) -> Dict[str, torch.Tensor]:
    result = {
        "input_ids": torch.randint(2, 1000, size=shape, dtype=torch.int64, device="cuda"),
        "attention_mask": get_attention_mask(shape).to(torch.int64),
    }

    if isinstance(model, BertPreTrainedModel):
        result["token_type_ids"] = torch.ones(size=shape, dtype=torch.int64, device="cuda")

    if isinstance(model, T5PreTrainedModel):
        result["decoder_input_ids"] = torch.randint(2, 1000, size=shape, dtype=torch.int64, device="cuda")

    return result
