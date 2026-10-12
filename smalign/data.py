"""Loading, filtering, and sampling the assignment datasets.

Every source is reduced to single-turn rows: ``{"id", "instruction", "response"}``
for SFT and ``{"id", "instruction", "chosen", "rejected"}`` for preferences.
Sampling shuffles a filtered index list with ``random.Random(seed)``, so the
subsets depend only on the source data and the seed, not on library versions.
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Iterable

from smalign.prompts import format_prompt

_HH_TURN = re.compile(r"\n\n(Human|Assistant): ")


def read_jsonl(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def sample(rows: list[dict], n: int, seed: int) -> list[dict]:
    order = list(range(len(rows)))
    random.Random(seed).shuffle(order)
    if len(order) < n:
        raise ValueError(f"only {len(order)} rows after filtering, need {n}")
    return [rows[i] for i in order[:n]]


def _n_tokens(tokenizer, texts: list[str]) -> list[int]:
    return [len(ids) for ids in tokenizer(texts, add_special_tokens=False)["input_ids"]]


# ---------------------------------------------------------------- SFT: UltraChat


def ultrachat_rows(split) -> list[dict]:
    """First user turn and first assistant reply of each conversation."""

    rows = []
    for ex in split:
        msgs = ex["messages"]
        if len(msgs) < 2 or msgs[0]["role"] != "user" or msgs[1]["role"] != "assistant":
            continue
        instruction, response = msgs[0]["content"].strip(), msgs[1]["content"].strip()
        if instruction and response:
            rows.append({"id": ex["prompt_id"], "instruction": instruction, "response": response})
    return rows


def keep_sft_fitting(rows: list[dict], tokenizer, max_length: int, max_prompt_length: int) -> list[dict]:
    """Keep rows whose formatted prompt, response, and EOS fit without truncation.

    Then every training target ends with EOS and the model learns when to stop.
    """

    prompt_len = _n_tokens(tokenizer, [format_prompt(r["instruction"]) for r in rows])
    resp_len = _n_tokens(tokenizer, [r["response"] for r in rows])
    return [
        r
        for r, p, a in zip(rows, prompt_len, resp_len)
        if p <= max_prompt_length and p + a + 1 <= max_length
    ]


# ---------------------------------------------------------------- DPO: HH-RLHF


def _hh_split(text: str) -> list[tuple[str, str]] | None:
    parts = _HH_TURN.split(text)
    if parts[0].strip():
        return None
    turns = [(parts[i], parts[i + 1].strip()) for i in range(1, len(parts) - 1, 2)]
    return turns or None


def hh_rows(split) -> list[dict]:
    """Single-turn HH-RLHF pairs: exactly one Human turn followed by one Assistant turn."""

    rows = []
    for i, ex in enumerate(split):
        chosen, rejected = _hh_split(ex["chosen"]), _hh_split(ex["rejected"])
        if not chosen or not rejected or len(chosen) != 2 or len(rejected) != 2:
            continue
        if [r for r, _ in chosen] != ["Human", "Assistant"] or [r for r, _ in rejected] != ["Human", "Assistant"]:
            continue
        if chosen[0][1] != rejected[0][1]:
            continue
        instruction, good, bad = chosen[0][1], chosen[1][1], rejected[1][1]
        if instruction and good and bad and good != bad:
            rows.append({"id": f"hh-{i}", "instruction": instruction, "chosen": good, "rejected": bad})
    return rows


# ---------------------------------------------------------------- DPO: UltraFeedback


def ultrafeedback_rows(split) -> list[dict]:
    """Binarized UltraFeedback pairs with a strict score preference."""

    rows = []
    for ex in split:
        if not ex["score_chosen"] > ex["score_rejected"]:
            continue
        chosen, rejected = ex["chosen"], ex["rejected"]
        if len(chosen) != 2 or len(rejected) != 2 or chosen[-1]["role"] != "assistant":
            continue
        instruction = ex["prompt"].strip()
        good, bad = chosen[-1]["content"].strip(), rejected[-1]["content"].strip()
        if instruction and good and bad and good != bad:
            rows.append({"id": ex["prompt_id"], "instruction": instruction, "chosen": good, "rejected": bad})
    return rows


def keep_short_prompts(rows: list[dict], tokenizer, max_prompt_length: int) -> list[dict]:
    """Drop pairs whose formatted prompt would need truncation; responses may still be cut at max_length."""

    prompt_len = _n_tokens(tokenizer, [format_prompt(r["instruction"]) for r in rows])
    return [r for r, p in zip(rows, prompt_len) if p <= max_prompt_length]


# ---------------------------------------------------------------- evaluation sets


def alpaca_eval_subset(n: int = 300, seed: int = 42) -> list[dict]:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download("tatsu-lab/alpaca_eval", "alpaca_eval.json", repo_type="dataset")
    with open(path, encoding="utf-8") as f:
        full = json.load(f)
    picked = sorted(random.Random(seed).sample(range(len(full)), n))
    return [
        {
            "id": f"alpaca-eval-{seed}-{i:04d}",
            "source_index": idx,
            "instruction": full[idx]["instruction"],
            "source": full[idx].get("dataset", ""),
            "dataset": "alpaca_eval",
        }
        for i, idx in enumerate(picked)
    ]
