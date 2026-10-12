"""Direct Preference Optimization with a from-scratch PyTorch loop (no TRL).

    python scripts/train_dpo.py --config configs/dpo_hh.yaml
    python scripts/train_dpo.py --config configs/dpo_uf.yaml

The policy and a frozen reference are both loaded from the SFT ``best``
checkpoint. With two GPUs the policy trains on cuda:0 and the reference runs on
cuda:1. Writes ``metrics.jsonl``, ``config.json``, ``summary.json``, and the
``best`` (highest validation reward accuracy) and ``final`` checkpoints.
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
    load_config,
    make_scaler,
    pick_device,
    save_checkpoint,
    set_seed,
)

DEFAULTS = {
    "sft_checkpoint": "outputs/sft/best",
    "train_path": "data/prepared/hh_train.jsonl",
    "val_path": "data/prepared/hh_val.jsonl",
    "out_dir": "outputs/dpo_hh",
    "max_length": 512,
    "max_prompt_length": 256,
    "batch_size": 1,
    "grad_accum": 32,
    "eval_batch_size": 4,
    "lr": 1e-6,
    "optimizer": "adamw",
    "weight_decay": 0.0,
    "beta": 0.1,
    "max_grad_norm": 1.0,
    "epochs": 1,
    "eval_every": 50,
    "log_every": 5,
    "fp16": True,
    "reference_dtype": "float32",
    "seed": 42,
    "max_train": None,
    "max_val": None,
    "time_limit_hours": None,
}


def pair_batches(rows, tokenizer, cfg, batch_size):
    for i in range(0, len(rows), batch_size):
        chunk = rows[i : i + batch_size]
        yield batching.build_dpo_batch(
            tokenizer,
            [format_prompt(r["instruction"]) for r in chunk],
            [r["chosen"] for r in chunk],
            [r["rejected"] for r in chunk],
            cfg["max_length"],
            cfg["max_prompt_length"],
        )


def pair_logps(model, dpo_batch, pad_id, device, fp16):
    """(chosen, rejected) summed response log-probs from one forward over both sides."""

    joined = batching.concat_pair_batch(dpo_batch, pad_id)
    with autocast(device, fp16):
        logps = losses.sequence_logprobs(model, joined)
    n = dpo_batch["chosen"]["input_ids"].shape[0]
    return logps[:n], logps[n:]


@torch.no_grad()
def reference_pair_logps(model, dpo_batch, pad_id, device, fp16):
    return pair_logps(model, dpo_batch, pad_id, device, fp16)


@torch.no_grad()
def evaluate(policy, val_batches, ref_val, pad_id, device, cfg) -> dict:
    policy.eval()
    totals = {"loss": 0.0, "acc": 0.0, "margin": 0.0, "chosen": 0.0, "rejected": 0.0}
    n = 0
    for dpo_batch, (ref_c, ref_r) in zip(val_batches, ref_val):
        pol_c, pol_r = pair_logps(policy, dpo_batch, pad_id, device, cfg["fp16"])
        loss, stats = losses.dpo_loss_from_logps(pol_c, pol_r, ref_c, ref_r, cfg["beta"])
        k = pol_c.shape[0]
        totals["loss"] += loss.item() * k
        totals["acc"] += stats["reward_correct"].sum().item()
        totals["margin"] += stats["reward_margin"].sum().item()
        totals["chosen"] += stats["chosen_rewards"].sum().item()
        totals["rejected"] += stats["rejected_rewards"].sum().item()
        n += k
    policy.train()
    return {
        "val_loss": totals["loss"] / n,
        "val_reward_acc": totals["acc"] / n,
        "val_reward_margin": totals["margin"] / n,
        "val_chosen_reward": totals["chosen"] / n,
        "val_rejected_reward": totals["rejected"] / n,
        "val_pairs": n,
    }


def main() -> None:
    cfg = load_config(DEFAULTS)
    set_seed(cfg["seed"])
    out = Path(cfg["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    policy_device = pick_device(0)
    ref_device = pick_device(1)

    tokenizer = AutoTokenizer.from_pretrained(cfg["sft_checkpoint"])
    pad_id = batching._pad_id(tokenizer)
    policy = AutoModelForCausalLM.from_pretrained(cfg["sft_checkpoint"], torch_dtype=torch.float32).to(policy_device)
    reference = AutoModelForCausalLM.from_pretrained(
        cfg["sft_checkpoint"], torch_dtype=getattr(torch, cfg["reference_dtype"])
    ).to(ref_device)
    for model in (policy, reference):
        model.config.use_cache = False
    policy.train()
    reference.eval()
    reference.requires_grad_(False)

    train_rows = read_jsonl(cfg["train_path"])[: cfg["max_train"]]
    val_rows = read_jsonl(cfg["val_path"])[: cfg["max_val"]]
    order = torch.randperm(len(train_rows), generator=torch.Generator().manual_seed(cfg["seed"])).tolist()

    if cfg["optimizer"] == "rmsprop":
        optimizer = torch.optim.RMSprop(policy.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    else:
        optimizer = torch.optim.AdamW(policy.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"],
                                      fused=policy_device.type == "cuda")
    scaler = make_scaler(policy_device, cfg["fp16"])

    n_micro = math.ceil(len(train_rows) / cfg["batch_size"])
    total_steps = math.ceil(n_micro / cfg["grad_accum"]) * cfg["epochs"]
    cfg.update(n_train=len(train_rows), n_val=len(val_rows), optimizer_steps=total_steps,
               effective_batch_size=cfg["batch_size"] * cfg["grad_accum"],
               policy_device=str(policy_device), reference_device=str(ref_device), torch=torch.__version__)
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    print(json.dumps(cfg, indent=2))

    # The reference is frozen, so its validation log-probs are computed once.
    val_batches = list(pair_batches(val_rows, tokenizer, cfg, cfg["eval_batch_size"]))
    ref_val = [
        tuple(x.to(policy_device) for x in reference_pair_logps(reference, b, pad_id, ref_device, cfg["fp16"]))
        for b in val_batches
    ]

    log = MetricsLog(out / "metrics.jsonl")
    best = {"val_reward_acc": -1.0, "val_reward_margin": -float("inf"), "step": -1}

    def run_eval(step: int) -> None:
        stats = evaluate(policy, val_batches, ref_val, pad_id, policy_device, cfg)
        key = (stats["val_reward_acc"], stats["val_reward_margin"])
        is_best = step > 0 and key > (best["val_reward_acc"], best["val_reward_margin"])
        log.write(kind="eval", step=step, **stats, is_best=is_best)
        print(f"[eval] step {step}/{total_steps} loss {stats['val_loss']:.4f} acc {stats['val_reward_acc']:.3f} "
              f"margin {stats['val_reward_margin']:.4f}" + (" *best*" if is_best else ""))
        if is_best:
            best.update(val_reward_acc=stats["val_reward_acc"], val_reward_margin=stats["val_reward_margin"], step=step)
            save_checkpoint(policy, tokenizer, out / "best", {"step": step, "reference": cfg["sft_checkpoint"], **stats})

    run_eval(0)
    deadline = time.time() + cfg["time_limit_hours"] * 3600 if cfg["time_limit_hours"] else None
    step, micro, stopped_early = 0, 0, False
    window = {"loss": 0.0, "acc": 0.0, "margin": 0.0, "chosen": 0.0, "rejected": 0.0, "pairs": 0}
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(cfg["epochs"]):
        epoch_rows = [train_rows[i] for i in order] if epoch == 0 else [
            train_rows[i] for i in torch.randperm(len(train_rows), generator=torch.Generator().manual_seed(cfg["seed"] + epoch)).tolist()
        ]
        for dpo_batch in pair_batches(epoch_rows, tokenizer, cfg, cfg["batch_size"]):
            # Launch the reference first: CUDA runs it on cuda:1 while the policy runs on cuda:0.
            ref_c, ref_r = reference_pair_logps(reference, dpo_batch, pad_id, ref_device, cfg["fp16"])
            pol_c, pol_r = pair_logps(policy, dpo_batch, pad_id, policy_device, cfg["fp16"])
            loss, stats = losses.dpo_loss_from_logps(pol_c, pol_r, ref_c, ref_r, cfg["beta"])
            scaler.scale(loss / cfg["grad_accum"]).backward()

            k = pol_c.shape[0]
            window["loss"] += loss.item() * k
            window["acc"] += stats["reward_correct"].sum().item()
            window["margin"] += stats["reward_margin"].sum().item()
            window["chosen"] += stats["chosen_rewards"].sum().item()
            window["rejected"] += stats["rejected_rewards"].sum().item()
            window["pairs"] += k
            micro += 1
            if micro % cfg["grad_accum"] and micro != n_micro * (epoch + 1):
                continue

            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg["max_grad_norm"])
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            step += 1

            if step % cfg["log_every"] == 0 or step == total_steps:
                p = window["pairs"]
                log.write(kind="train", step=step, epoch=epoch, train_loss=window["loss"] / p,
                          train_reward_acc=window["acc"] / p, train_reward_margin=window["margin"] / p,
                          train_chosen_reward=window["chosen"] / p, train_rejected_reward=window["rejected"] / p,
                          grad_norm=float(grad_norm), lr=optimizer.param_groups[0]["lr"],
                          loss_scale=scaler.get_scale() if scaler.is_enabled() else 1.0)
                print(f"step {step}/{total_steps} loss {window['loss'] / p:.4f} acc {window['acc'] / p:.3f} "
                      f"margin {window['margin'] / p:.4f} gnorm {float(grad_norm):.2f}")
                window = dict.fromkeys(window, 0.0)
                window["pairs"] = 0

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
    save_checkpoint(policy, tokenizer, out / "final", {"step": step, "reference": cfg["sft_checkpoint"]})
    summary = {"best_step": best["step"], "best_val_reward_acc": best["val_reward_acc"],
               "best_val_reward_margin": best["val_reward_margin"], "final_step": step,
               "total_steps": total_steps, "stopped_early": stopped_early}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
