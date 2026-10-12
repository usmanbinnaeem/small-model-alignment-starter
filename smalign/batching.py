"""Tokenization into response-masked causal-LM batches.

Prompt and response are tokenized separately and concatenated, so the prompt
length in tokens is exact and the boundary never merges across the two. Labels
are aligned with ``input_ids`` (the shift happens in the loss), with prompt and
padding positions set to ``IGNORE_INDEX``. Padding is masked by position, not
by token id, because many tokenizers (Qwen2.5 included) use EOS as PAD.
"""

from __future__ import annotations

from typing import Sequence

import torch

IGNORE_INDEX = -100


def _encode(tokenizer, text: str) -> list[int]:
    return list(tokenizer.encode(text, add_special_tokens=False))


def _pad_id(tokenizer) -> int:
    if getattr(tokenizer, "pad_token_id", None) is not None:
        return tokenizer.pad_token_id
    if getattr(tokenizer, "eos_token_id", None) is not None:
        return tokenizer.eos_token_id
    return 0


def tokenize_example(
    tokenizer,
    prompt: str,
    response: str,
    max_length: int,
    max_prompt_length: int | None = None,
) -> tuple[list[int], list[int]]:
    """Return (input_ids, labels) for one row, before padding.

    EOS is appended to the response before truncation, so a response cut at
    ``max_length`` loses its EOS. ``max_prompt_length`` keeps the *last* prompt
    tokens so the assistant header that precedes the response survives.
    """

    prompt_ids = _encode(tokenizer, prompt)
    if max_prompt_length is not None and len(prompt_ids) > max_prompt_length:
        prompt_ids = prompt_ids[-max_prompt_length:] if max_prompt_length > 0 else []

    response_ids = _encode(tokenizer, response)
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if eos_id is not None:
        response_ids = response_ids + [eos_id]

    input_ids = (prompt_ids + response_ids)[:max_length]
    labels = ([IGNORE_INDEX] * len(prompt_ids) + response_ids)[:max_length]
    return input_ids, labels


def collate(rows: Sequence[tuple[list[int], list[int]]], pad_id: int) -> dict[str, torch.Tensor]:
    """Right-pad (input_ids, labels) rows to a common width, keeping row order."""

    width = max(1, max(len(ids) for ids, _ in rows))
    input_ids = torch.full((len(rows), width), pad_id, dtype=torch.long)
    labels = torch.full((len(rows), width), IGNORE_INDEX, dtype=torch.long)
    attention_mask = torch.zeros((len(rows), width), dtype=torch.long)
    for i, (ids, lab) in enumerate(rows):
        n = len(ids)
        if n:
            input_ids[i, :n] = torch.tensor(ids, dtype=torch.long)
            labels[i, :n] = torch.tensor(lab, dtype=torch.long)
            attention_mask[i, :n] = 1
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def build_lm_batch(
    tokenizer,
    prompts: Sequence[str],
    responses: Sequence[str],
    max_length: int,
    max_prompt_length: int | None = None,
) -> dict[str, torch.Tensor]:
    if len(prompts) != len(responses):
        raise ValueError(f"{len(prompts)} prompts but {len(responses)} responses")
    rows = [
        tokenize_example(tokenizer, p, r, max_length, max_prompt_length)
        for p, r in zip(prompts, responses)
    ]
    return collate(rows, _pad_id(tokenizer))


def build_dpo_batch(
    tokenizer,
    prompts: Sequence[str],
    chosen: Sequence[str],
    rejected: Sequence[str],
    max_length: int,
    max_prompt_length: int | None = None,
) -> dict[str, dict[str, torch.Tensor]]:
    if not (len(prompts) == len(chosen) == len(rejected)):
        raise ValueError("prompts, chosen, and rejected must have the same length")
    return {
        "chosen": build_lm_batch(tokenizer, prompts, chosen, max_length, max_prompt_length),
        "rejected": build_lm_batch(tokenizer, prompts, rejected, max_length, max_prompt_length),
    }


def concat_pair_batch(dpo_batch: dict[str, dict[str, torch.Tensor]], pad_id: int) -> dict[str, torch.Tensor]:
    """Stack chosen rows over rejected rows (padded to one width) for a single forward pass."""

    chosen, rejected = dpo_batch["chosen"], dpo_batch["rejected"]
    width = max(chosen["input_ids"].shape[1], rejected["input_ids"].shape[1])

    def pad(t: torch.Tensor, value: int) -> torch.Tensor:
        extra = width - t.shape[1]
        if extra == 0:
            return t
        return torch.cat([t, t.new_full((t.shape[0], extra), value)], dim=1)

    return {
        "input_ids": torch.cat([pad(chosen["input_ids"], pad_id), pad(rejected["input_ids"], pad_id)]),
        "attention_mask": torch.cat([pad(chosen["attention_mask"], 0), pad(rejected["attention_mask"], 0)]),
        "labels": torch.cat([pad(chosen["labels"], IGNORE_INDEX), pad(rejected["labels"], IGNORE_INDEX)]),
    }
