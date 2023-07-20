# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.
import argparse

from utils.config import Config
from llama import Llama


def main(
        ckpt_dir: str,
        tokenizer_path: str,
        temperature: float,
        top_p: float,
        max_seq_len: int,
        max_gen_len: int,
        max_batch_size: int,
):
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    # prompts = [
    # For these prompts, the expected answer is the natural continuation of the prompt
    # "I believe the meaning of life is",
    # "Simply put, the theory of relativity states that ",
    # """A brief message congratulating the team on the launch:
    #
    # Hi everyone,
    #
    # I just """,
    # # Few shot prompt (providing a few examples before asking model to complete more);
    # """Translate English to French:
    #
    # sea otter => loutre de mer
    # peppermint => menthe poivrée
    # plush girafe => girafe peluche
    # cheese =>""",
    # ]
    prompts = ["I believe the meaning of life is"] * args.max_batch_size
    for _ in range(2):  # warmup
        results, batched_token_timings = generator.text_completion(
            prompts,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        )

    for prompt, result in zip(prompts, results):
        print(prompt)
        print(f"> {result['generation']}")
        print("\n==================================\n")
    print(f"batch size: {len(prompts)}")
    print(f"longer generated sequence in the batch: {len(batched_token_timings)}")
    print(f"total inference time (fwd): {sum(batched_token_timings):.2f}")
    print(f"average token timings: {sum(batched_token_timings) / len(batched_token_timings):.2f}")
    print(f"min/max token timings: {min(batched_token_timings):.2f}/{max(batched_token_timings):.2f}")
    print(f"token / sec: {max_batch_size * len(batched_token_timings) / sum(batched_token_timings):.2f}")


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()

    argparser.add_argument(
        "--ckpt_dir",
        type=str,
        default="./llama-2-7b",
        help="Path to the checkpoint directory",
    )
    argparser.add_argument(
        "--tokenizer_path",
        type=str,
        default="./tokenizer.model",
        help="Path to the tokenizer model",
    )
    argparser.add_argument(
        "--temperature",
        type=float,
        default=0.6,
        help="Temperature for sampling",
    )
    argparser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Top p for sampling",
    )
    argparser.add_argument(
        "--max_seq_len",
        type=int,
        default=512,
        help="Maximum sequence length",
    )
    argparser.add_argument(
        "--max_gen_len",
        type=int,
        default=512,
        help="Maximum generation length",
    )
    argparser.add_argument(
        "--max_batch_size",
        type=int,
        default=1,
        help="Maximum batch size",
    )
    argparser.add_argument(
        "--enable_nvtx",
        action="store_true",
        help="Enable NVTX profiling",
    )
    # enable use_triton in config
    argparser.add_argument(
        "--use_triton",
        action="store_true",
        help="Use Triton kernels instead of PyTorch implementation",
    )

    args = argparser.parse_args()

    config = Config()
    config.set_nvtx(args.enable_nvtx)
    config.set_use_triton(args.use_triton)

    main(
        ckpt_dir=args.ckpt_dir,
        tokenizer_path=args.tokenizer_path,
        temperature=args.temperature,
        top_p=args.top_p,
        max_seq_len=args.max_seq_len,
        max_gen_len=args.max_gen_len,
        max_batch_size=args.max_batch_size,
    )
