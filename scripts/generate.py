"""Generate responses for a fixed evaluation set.

    python scripts/generate.py --model outputs/sft/best --generator sft \
        --eval_set eval_sets/alpaca_eval_300_seed42.json --out outputs/gen/alpaca_sft.json

Decoding is greedy by default, with the same prompt format used in training:
the fixed ChatML prompt from ``smalign.prompts`` for the Qwen runs, or, with
``--chat_template``, the model's own chat template (the persona LoRA task,
where TRL formats training data with that template).
The output file is rewritten after every batch, and rows already in it are
skipped, so an interrupted run resumes where it stopped.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from smalign.prompts import format_prompt
from smalign.train_utils import pick_device

PASSTHROUGH = ("category", "expected_behavior", "source")
STOP_STRINGS = ("<|im_end|>", "<|im_start|>", "<|endoftext|>")


def write_atomic(path: Path, rows: list[dict]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--generator", required=True, help="name stored in each row, e.g. qwen2.5-0.5b-base")
    ap.add_argument("--eval_set", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dataset", default=None, help="defaults to each row's dataset field")
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--repetition_penalty", type=float, default=1.0)
    ap.add_argument("--no_repeat_ngram_size", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--chat_template", action="store_true", help="format prompts with the tokenizer's chat template")
    args = ap.parse_args()

    rows = json.load(open(args.eval_set, encoding="utf-8"))[: args.limit]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {r["id"]: r for r in json.load(open(out, encoding="utf-8"))} if out.exists() else {}
    todo = [r for r in rows if r["id"] not in done]
    print(f"{len(done)} done, {len(todo)} to generate -> {out}")

    device = pick_device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(device).eval()

    stop_ids = sorted({tokenizer.eos_token_id, *[tokenizer.convert_tokens_to_ids(t) for t in ("<|im_end|>",)]} - {None})
    settings = {
        "do_sample": False,
        "max_new_tokens": args.max_new_tokens,
        "repetition_penalty": args.repetition_penalty,
        "no_repeat_ngram_size": args.no_repeat_ngram_size,
        "eos_token_id": stop_ids,
        "pad_token_id": tokenizer.pad_token_id,
    }
    prompt_format = "tokenizer chat template" if args.chat_template else "smalign.prompts.format_prompt (ChatML)"
    (out.parent / f"{out.stem}.settings.json").write_text(
        json.dumps({"model": args.model, "prompt_format": prompt_format, **settings}, indent=2))

    # Batch similar lengths together; output order follows the evaluation set.
    todo.sort(key=lambda r: len(r["instruction"]))
    for start in range(0, len(todo), args.batch_size):
        chunk = todo[start : start + args.batch_size]
        if args.chat_template:
            texts = [tokenizer.apply_chat_template([{"role": "user", "content": r["instruction"]}],
                                                   tokenize=False, add_generation_prompt=True) for r in chunk]
        else:
            texts = [format_prompt(r["instruction"]) for r in chunk]
        enc = tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        with torch.no_grad():
            gen = model.generate(**enc, **{k: v for k, v in settings.items() if v or k == "do_sample"})
        new_tokens = gen[:, enc["input_ids"].shape[1] :]
        for r, toks in zip(chunk, new_tokens):
            toks = toks.tolist()
            hit_stop = any(t in stop_ids for t in toks)
            text = tokenizer.decode(toks, skip_special_tokens=False)
            for s in STOP_STRINGS:
                text = text.split(s)[0]
            done[r["id"]] = {
                "id": r["id"],
                "instruction": r["instruction"],
                "output": text.strip(),
                "generator": args.generator,
                "dataset": args.dataset or r.get("dataset", ""),
                "finish_reason": "stop" if hit_stop else "length",
                **{k: r[k] for k in PASSTHROUGH if k in r},
            }
        write_atomic(out, [done[r["id"]] for r in rows if r["id"] in done])
        print(f"  {min(start + args.batch_size, len(todo))}/{len(todo)}")

    final = [done[r["id"]] for r in rows]
    write_atomic(out, final)
    n_len = sum(r["finish_reason"] == "length" for r in final)
    print(f"wrote {len(final)} rows; {n_len} hit max_new_tokens")


if __name__ == "__main__":
    main()
