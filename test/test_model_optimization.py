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
from typing import Tuple

import pytest
import torch
from transformers import AutoModelForSequenceClassification

from kernl.model_optimization import optimize_model


model_name = "BaptisteDoyen/camembert-base-xnli"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model = model.eval().cuda()


@pytest.mark.parametrize("shape", [(1, w) for w in range(8, 128 + 8, 8)], ids=lambda s: "x".join(map(str, s)))
def test_original_model(shape: Tuple[int, int]):
    with torch.inference_mode(), torch.cuda.amp.autocast(enabled=True, dtype=torch.float16, cache_enabled=True):
        inputs = {
            "input_ids": torch.ones(shape, device="cuda", dtype=torch.long),
            "attention_mask": torch.ones(shape, device="cuda", dtype=torch.long),
        }
        _ = model(**inputs)


optimized_model = optimize_model(model)


@pytest.mark.parametrize("shape", [(1, w) for w in range(8, 128 + 8, 8)], ids=lambda s: "x".join(map(str, s)))
def test_optimized_model(shape: Tuple[int, int]):
    with torch.inference_mode(), torch.cuda.amp.autocast(enabled=True, dtype=torch.float16, cache_enabled=True):
        inputs = {
            "input_ids": torch.ones(shape, device="cuda", dtype=torch.long),
            "attention_mask": torch.ones(shape, device="cuda", dtype=torch.long),
        }
        _ = optimized_model(**inputs)

        # assert that the original model can not be used after otpimization:
        with pytest.raises(Exception) as exc_info:
            _ = model(**inputs)
        assert exc_info.value.args[0] == "Original model can not be used after optimization"
