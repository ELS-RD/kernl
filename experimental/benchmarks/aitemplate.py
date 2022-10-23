#  Copyright (c) Meta Platforms, Inc. and affiliates.
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
import time

import click
import torch
from benchmark_ait import compile_module
from modeling.torch_model import BertBaseUncased as BertPt


def run_model(activation: str, graph_mode: bool, use_fp16_acc: bool, verify: bool):
    f = open("measures.txt", mode="w")
    for shape in [(bs, seq_l) for bs in [1, 8, 32] for seq_l in [16, 32, 64, 128, 256, 384, 512] if bs * seq_l < 10000]:
        inputs_pt = {
            "input_ids": torch.randint(2, 1000, size=shape, dtype=torch.int64, device="cuda"),
            "position_ids": torch.arange(shape[1], dtype=torch.int64).expand(shape).contiguous().cuda(),
            "token_type_ids": torch.ones(size=shape, dtype=torch.int64, device="cuda"),
        }

        batch_size, seq_len = inputs_pt["input_ids"].size()

        pt_model = BertPt(pretrained=True)._model
        pt_model.eval()
        hidden_size = pt_model.config.hidden_size

        mod = compile_module(batch_size, seq_len, hidden_size, activation, use_fp16_acc, False, pt_model)

        outputs = [torch.empty(mod.get_output_maximum_shape(0)).half().cuda()]

        # warmup
        for _ in range(10):
            mod.run_with_tensors(inputs_pt, outputs, graph_mode=graph_mode)

        torch.cuda.synchronize()
        timings = list()
        for _ in range(10):
            start = time.perf_counter()
            mod.run_with_tensors(inputs_pt, outputs, graph_mode=graph_mode)
            torch.cuda.synchronize()
            end = time.perf_counter()
            timings.append(end - start)

        f.write(f"{shape}: {torch.median(torch.tensor(timings)):.4f}\n")
        f.flush()
        print(f"Logits: {outputs[0]}")
        if verify:
            pt_outputs = pt_model.bert(**inputs_pt)
            torch.allclose(outputs[0], pt_outputs.last_hidden_state, 1e-1, 1e-1)
            print("Verification done!")
    f.close()


@click.command()
@click.option(
    "--activation",
    type=str,
    default="gelu",
    help="Activation function applied on BERT, currently only support gelu and fast_gelu",
)
@click.option(
    "--graph_mode",
    type=bool,
    default=True,
    help="Use CUDA graph or not. (hipGraph is not supported yet)",
)
@click.option(
    "--use_fp16_acc",
    type=bool,
    default=False,
    help="Use fp16 accumulation or not (TensorRT is using fp16_acc)",
)
@click.option(
    "--verify",
    type=bool,
    default=True,
    help="Verify AIT outputs against PT",
)
def run_demo(
    activation: str,
    graph_mode: bool,
    use_fp16_acc: bool,
    verify: bool,
):
    run_model(activation, graph_mode, use_fp16_acc, verify)


if __name__ == "__main__":
    torch.manual_seed(4896)
    run_demo()
