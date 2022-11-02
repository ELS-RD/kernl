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
from transformers import AutoModel

from conftest import set_seed

from kernl.model_optimization import optimize_model


@set_seed()
def test_optimized_model():
    shape = (1, 128)
    model_name = "bert-base-uncased"
    model = AutoModel.from_pretrained(model_name).eval().cuda()
    optimized_model = AutoModel.from_pretrained(model_name).eval().cuda()
    optimize_model(optimized_model)

    with torch.inference_mode(), torch.cuda.amp.autocast(enabled=True, dtype=torch.float16, cache_enabled=True):
        inputs = {
            "input_ids": torch.randint(2, 10000, shape, device="cuda", dtype=torch.long),
            "attention_mask": torch.ones(shape, device="cuda", dtype=torch.long),
        }
        output_pytorch = model(**inputs)
        output_optimized = optimized_model(**inputs)

        assert torch.allclose(output_pytorch.last_hidden_state, output_optimized.last_hidden_state.float(), atol=1e-1)
