"""Supervised fine-tuning with a from-scratch PyTorch loop (no HF Trainer).

    python scripts/train_sft.py --config configs/sft.yaml
    python scripts/train_sft.py --config configs/sft.yaml --max_train 64 --eval_every 2   # smoke test

Writes ``metrics.jsonl``, ``config.json``, ``summary.json``, and the ``best``
(lowest validation loss) and ``final`` checkpoints under ``--out_dir``.

Resumable: every ``save_every`` optimizer steps the model, optimizer,
scheduler, grad scaler, and data position are saved to ``out_dir/resume``.
Rerunning with the same ``out_dir`` continues from there, and a run whose
``summary.json`` says it finished exits immediately.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from smalign import batching, losses
from smalign.data import read_jsonl
from smalign.prompts import format_prompt
from smalign.train_utils import (
    MetricsLog,
    autocast,
    batch_order,
    decay_groups,
    drop_resume,
    find_resume,
    load_config,
    load_resume,
    make_scaler,
    pick_device,
    save_checkpoint,
    save_resume,
    set_seed,
    trim_metrics,
    update_resume_state,
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
    "save_every": 100,
    "log_every": 10,
    "fp16": True,
    "seed": 42,
    "max_train": None,
    "max_val": None,
    "time_limit_hours": None,
    "keep_resume": False,
}


def tokenize_rows(rows, tokenizer, max_length):
    return [
        batching.tokenize_example(tokenizer, format_prompt(r["instruction"]), r["response"], max_length)
        for r in rows
    ]


@torch.no_grad()
def evaluate(model, val_data, batch_size, pad_id, device, fp16) -> dict:
    model.eval()
    total_nll, total_tokens = 0.0, 0
    for i in range(0, len(val_data), batch_size):
        batch = batching.collate(val_data[i : i + batch_size], pad_id)
        with autocast(device, fp16):
            loss, n = losses.sft_loss(model, batch)
        total_nll += loss.item() * n
        total_tokens += n
    model.train()
    mean = total_nll / max(total_tokens, 1)
    return {"val_loss": mean, "val_ppl": math.exp(min(mean, 50)), "val_tokens": total_tokens}


def main() -> None:
    cfg = load_config(DEFAULTS)
    out = Path(cfg["out_dir"])
    out.mkdir(parents=True, exist_ok=True)

    summary_path = out / "summary.json"
    if summary_path.exists() and not json.loads(summary_path.read_text()).get("stopped_early"):
        print(f"{summary_path} says this run finished; nothing to do.")
        return

    set_seed(cfg["seed"])
    device = pick_device(0)
    resume = find_resume(out)

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"])
    model_path = resume / "model" if resume else cfg["model"]
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32).to(device)
    model.config.use_cache = False
    model.train()

    train_rows = read_jsonl(cfg["train_path"])[: cfg["max_train"]]
    val_rows = read_jsonl(cfg["val_path"])[: cfg["max_val"]]
    pad_id = batching._pad_id(tokenizer)
    train_data = tokenize_rows(train_rows, tokenizer, cfg["max_length"])
    val_data = tokenize_rows(val_rows, tokenizer, cfg["max_length"])

    micro_per_epoch = math.ceil(len(train_data) / cfg["batch_size"])
    steps_per_epoch = math.ceil(micro_per_epoch / cfg["grad_accum"])
    total_steps = steps_per_epoch * cfg["epochs"]
    optimizer = torch.optim.AdamW(decay_groups(model, cfg["weight_decay"]), lr=cfg["lr"],
                                  fused=device.type == "cuda")
    scheduler = warmup_cosine(optimizer, total_steps, cfg["warmup_ratio"])
    scaler = make_scaler(device, cfg["fp16"])

    cfg.update(n_train=len(train_data), n_val=len(val_data), optimizer_steps=total_steps,
               effective_batch_size=cfg["batch_size"] * cfg["grad_accum"], device=str(device),
               loss_reduction="mean NLL over all response tokens in the micro-batch",
               torch=torch.__version__)

    # Position in training: optimizer steps done, and micro-batches consumed in the current epoch.
    pos = {"step": 0, "epoch": 0, "micro_in_epoch": 0}
    best = {"val_loss": float("inf"), "step": -1}
    if resume:
        saved = load_resume(resume, optimizer, scheduler, scaler)
        pos, best = saved["pos"], saved["best"]
        trim_metrics(out / "metrics.jsonl", pos["step"])
        print(f"resumed from {resume} at step {pos['step']}/{total_steps}")
        cfg.setdefault("resumed_at_steps", []).append(pos["step"])
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    print(json.dumps(cfg, indent=2))

    log = MetricsLog(out / "metrics.jsonl")
    if resume:
        log.write(kind="resume", step=pos["step"])

    def run_eval(step: int) -> None:
        stats = evaluate(model, val_data, cfg["eval_batch_size"], pad_id, device, cfg["fp16"])
        # Step 0 is the untrained base model, so it is logged but never saved as best.
        is_best = step > 0 and stats["val_loss"] < best["val_loss"]
        log.write(kind="eval", step=step, **stats, is_best=is_best)
        print(f"[eval] step {step}/{total_steps} val_loss {stats['val_loss']:.4f}" + (" *best*" if is_best else ""))
        if is_best:
            best.update(val_loss=stats["val_loss"], step=step)
            save_checkpoint(model, tokenizer, out / "best", {"step": step, **stats})

    def checkpoint_progress() -> None:
        save_resume(out, model, optimizer, scheduler, scaler, {"pos": pos, "best": best})
        print(f"[resume] saved at step {pos['step']}")

    if not resume:
        run_eval(0)
    deadline = time.time() + cfg["time_limit_hours"] * 3600 if cfg["time_limit_hours"] else None
    window_loss, window_micro, window_tokens, t0 = 0.0, 0, 0, time.time()
    stopped_early = False
    optimizer.zero_grad(set_to_none=True)

    while pos["epoch"] < cfg["epochs"] and not stopped_early:
        batches = batch_order(len(train_data), cfg["batch_size"], cfg["seed"], pos["epoch"])
        for idx in batches[pos["micro_in_epoch"] :]:
            batch = batching.collate([train_data[i] for i in idx], pad_id)
            with autocast(device, cfg["fp16"]):
                loss, n_tokens = losses.sft_loss(model, batch)
            scaler.scale(loss / cfg["grad_accum"]).backward()
            window_loss += loss.item()
            window_micro += 1
            window_tokens += n_tokens
            pos["micro_in_epoch"] += 1
            end_of_epoch = pos["micro_in_epoch"] == micro_per_epoch
            if pos["micro_in_epoch"] % cfg["grad_accum"] and not end_of_epoch:
                continue

            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["max_grad_norm"])
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            pos["step"] += 1
            step = pos["step"]
            if end_of_epoch:
                pos["epoch"] += 1
                pos["micro_in_epoch"] = 0

            if step % cfg["log_every"] == 0 or step == total_steps:
                elapsed = time.time() - t0
                log.write(kind="train", step=step, epoch=pos["epoch"], train_loss=window_loss / window_micro,
                          lr=scheduler.get_last_lr()[0], grad_norm=float(grad_norm),
                          loss_scale=scaler.get_scale() if scaler.is_enabled() else 1.0,
                          tokens_per_s=window_tokens / max(elapsed, 1e-9))
                print(f"step {step}/{total_steps} loss {window_loss / window_micro:.4f} "
                      f"lr {scheduler.get_last_lr()[0]:.2e} gnorm {float(grad_norm):.2f}", flush=True)
                window_loss, window_micro, window_tokens, t0 = 0.0, 0, 0, time.time()

            # Save progress before evaluating, so a failure during eval cannot lose training steps.
            out_of_time = deadline is not None and time.time() > deadline and step != total_steps
            should_save = (step % cfg["save_every"] == 0 or out_of_time) and step != total_steps
            if should_save:
                checkpoint_progress()
            if step % cfg["eval_every"] == 0 or step == total_steps:
                run_eval(step)
                if should_save:
                    update_resume_state(out, {"pos": pos, "best": best})
            if out_of_time:
                print("time limit reached; rerun to continue from the saved step")
                stopped_early = True
                break

    step = pos["step"]
    if not stopped_early:
        if step % cfg["eval_every"] and step != total_steps:
            run_eval(step)
        save_checkpoint(model, tokenizer, out / "final", {"step": step})
        if not cfg["keep_resume"]:
            drop_resume(out)
    summary = {"best_step": best["step"], "best_val_loss": best["val_loss"], "final_step": step,
               "total_steps": total_steps, "stopped_early": stopped_early}
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
