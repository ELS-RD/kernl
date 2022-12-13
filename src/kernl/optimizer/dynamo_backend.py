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

from kernl.optimizer.attention import fuse_attention_pattern_1, fuse_attention_pattern_2, fuse_attention_pattern_3
from kernl.optimizer.dropout import remove_dropout
from kernl.optimizer.layer_norm import replace_layer_norm, replace_layer_norm_rms
from kernl.optimizer.linear import replace_all_linear
from kernl.optimizer.normalizer import normalize_operators


def dynamo_backend_ofi(gm: torch.fx.GraphModule, assume_causal=False):
    normalize_operators(gm)
    remove_dropout(gm)
    fuse_attention_pattern_1(gm, assume_causal)
    fuse_attention_pattern_2(gm, assume_causal)
    fuse_attention_pattern_3(gm, assume_causal)
    replace_all_linear(gm)
    replace_layer_norm(gm)
    replace_layer_norm_rms(gm)
    return gm
