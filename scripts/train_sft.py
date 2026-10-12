"""Supervised fine-tuning with a from-scratch PyTorch loop (no HF Trainer).

    python scripts/train_sft.py --config configs/sft.yaml
    python scripts/train_sft.py --config configs/sft.yaml --max_train 64 --eval_every 2   # smoke test

Writes ``metrics.jsonl``, ``config.json``, ``summary.json``, and the ``best``
(lowest validation loss) and ``final`` checkpoints under ``--out_dir``.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from smalign import batching, losses
from smalign.data import read_jsonl
from smalign.prompts import format_prompt
from smalign.train_utils import (
    MetricsLog,
    autocast,
    decay_groups,
    load_config,
    make_scaler,
    pick_device,
    save_checkpoint,
    set_seed,
    warmup_cosine,
)

DEFAULTS = {
    "model": "Qwen/Qwen2.5-0.5B",
    "train_path": "data/prepared/sft_train.jsonl",
    "val_path": "data/prepared/sft_val.jsonl",
    "out_dir": "outputs/sft",
    "max_length": 512,
    "batch_size": 2,
    "grad_accum": 16,
    "eval_batch_size": 8,
    "lr": 2e-5,
    "warmup_ratio": 0.03,
    "weight_decay": 0.1,
    "max_grad_norm": 1.0,
    "epochs": 1,
    "eval_every": 100,
    "log_every": 10,
    "fp16": True,
    "seed": 42,
    "max_train": None,
    "max_val": None,
    "time_limit_hours": None,
}


def tokenize_rows(rows, tokenizer, max_length):
    return [
        batching.tokenize_example(tokenizer, format_prompt(r["instruction"]), r["response"], max_length)
        for r in rows
    ]


@torch.no_grad()
def evaluate(model, loader, device, fp16) -> dict:
    model.eval()
    total_nll, total_tokens = 0.0, 0
    for batch in loader:
        with autocast(device, fp16):
            loss, n = losses.sft_loss(model, batch)
        total_nll += loss.item() * n
        total_tokens += n
    model.train()
    mean = total_nll / max(total_tokens, 1)
    return {"val_loss": mean, "val_ppl": math.exp(min(mean, 50)), "val_tokens": total_tokens}


def main() -> None:
    cfg = load_config(DEFAULTS)
    set_seed(cfg["seed"])
    out = Path(cfg["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    device = pick_device(0)

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"])
    model = AutoModelForCausalLM.from_pretrained(cfg["model"], torch_dtype=torch.float32).to(device)
    model.config.use_cache = False
    model.train()

    train_rows = read_jsonl(cfg["train_path"])[: cfg["max_train"]]
    val_rows = read_jsonl(cfg["val_path"])[: cfg["max_val"]]
    pad_id = batching._pad_id(tokenizer)
    train_data = tokenize_rows(train_rows, tokenizer, cfg["max_length"])
    val_data = tokenize_rows(val_rows, tokenizer, cfg["max_length"])

    collate = lambda rows: batching.collate(rows, pad_id)  # noqa: E731
    generator = torch.Generator().manual_seed(cfg["seed"])
    train_loader = DataLoader(train_data, batch_size=cfg["batch_size"], shuffle=True,
                              generator=generator, collate_fn=collate, drop_last=False)
    val_loader = DataLoader(val_data, batch_size=cfg["eval_batch_size"], shuffle=False, collate_fn=collate)

    steps_per_epoch = math.ceil(len(train_loader) / cfg["grad_accum"])
    total_steps = steps_per_epoch * cfg["epochs"]
    optimizer = torch.optim.AdamW(decay_groups(model, cfg["weight_decay"]), lr=cfg["lr"],
                                  fused=device.type == "cuda")
    scheduler = warmup_cosine(optimizer, total_steps, cfg["warmup_ratio"])
    scaler = make_scaler(device, cfg["fp16"])

    cfg.update(n_train=len(train_data), n_val=len(val_data), optimizer_steps=total_steps,
               effective_batch_size=cfg["batch_size"] * cfg["grad_accum"], device=str(device),
               loss_reduction="mean NLL over all response tokens in the micro-batch",
               torch=torch.__version__)
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    print(json.dumps(cfg, indent=2))

    log = MetricsLog(out / "metrics.jsonl")
    best = {"val_loss": float("inf"), "step": -1}

    def run_eval(step: int) -> None:
        stats = evaluate(model, val_loader, device, cfg["fp16"])
        # Step 0 is the untrained base model, so it is logged but never saved as best.
        is_best = step > 0 and stats["val_loss"] < best["val_loss"]
        log.write(kind="eval", step=step, **stats, is_best=is_best)
        print(f"[eval] step {step}/{total_steps} val_loss {stats['val_loss']:.4f}" + (" *best*" if is_best else ""))
        if is_best:
            best.update(val_loss=stats["val_loss"], step=step)
            save_checkpoint(model, tokenizer, out / "best", {"step": step, **stats})

    run_eval(0)
    deadline = time.time() + cfg["time_limit_hours"] * 3600 if cfg["time_limit_hours"] else None
    step, micro, window_loss, window_tokens, t0 = 0, 0, 0.0, 0, time.time()
    stopped_early = False
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(cfg["epochs"]):
        for batch in train_loader:
            with autocast(device, cfg["fp16"]):
                loss, n_tokens = losses.sft_loss(model, batch)
            scaler.scale(loss / cfg["grad_accum"]).backward()
            window_loss += loss.item()
            window_tokens += n_tokens
            micro += 1
            last_micro = micro == len(train_loader) * (epoch + 1)
            if micro % cfg["grad_accum"] and not last_micro:
                continue

            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["max_grad_norm"])
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1

            if step % cfg["log_every"] == 0 or step == total_steps:
                n_micro = cfg["grad_accum"] * cfg["log_every"]
                elapsed = time.time() - t0
                log.write(kind="train", step=step, epoch=epoch, train_loss=window_loss / n_micro,
                          lr=scheduler.get_last_lr()[0], grad_norm=float(grad_norm),
                          loss_scale=scaler.get_scale() if scaler.is_enabled() else 1.0,
                          tokens_per_s=window_tokens / max(elapsed, 1e-9))
                print(f"step {step}/{total_steps} loss {window_loss / n_micro:.4f} "
                      f"lr {scheduler.get_last_lr()[0]:.2e} gnorm {float(grad_norm):.2f}")
                window_loss, window_tokens, t0 = 0.0, 0, time.time()

            if step % cfg["eval_every"] == 0 or step == total_steps:
                run_eval(step)
            if deadline and time.time() > deadline:
                print("time limit reached, stopping")
                stopped_early = True
                break
        if stopped_early:
            break

    if step % cfg["eval_every"] and step != total_steps:
        run_eval(step)
    save_checkpoint(model, tokenizer, out / "final", {"step": step})
    summary = {"best_step": best["step"], "best_val_loss": best["val_loss"], "final_step": step,
               "total_steps": total_steps, "stopped_early": stopped_early}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
