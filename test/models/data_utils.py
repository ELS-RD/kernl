from typing import Dict

import torch


def get_attention_mask(shape: (int, int)) -> torch.Tensor:
    return torch.randint(1, shape[1], (shape[0],), device="cuda")[:, None] > torch.arange(0, shape[1], device="cuda")[
                                                                             None, :]

def get_input_causal(shape: (int, int)) -> Dict[str, torch.Tensor]:
    batch, seq_length = shape
    mask = torch.tril(torch.ones((batch, seq_length, seq_length), dtype=torch.int64, device="cuda"))
    return {
        "input_ids": torch.randint(2, 1000, size=shape, dtype=torch.int64, device="cuda"),
        "attention_mask": mask,
        "token_type_ids": torch.ones(size=shape, dtype=torch.int64, device="cuda")
    }


def get_input_non_causal(shape: (int, int)) -> Dict[str, torch.Tensor]:
    return {
        "input_ids": torch.randint(2, 1000, size=shape, dtype=torch.int64, device="cuda"),
        "attention_mask": get_attention_mask(shape).to(torch.int64),
        "token_type_ids": torch.ones(size=shape, dtype=torch.int64, device="cuda")
    }
