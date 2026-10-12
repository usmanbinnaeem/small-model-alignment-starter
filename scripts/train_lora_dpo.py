"""Task 5: persona (Ramsay-style) DPO with TRL's DPOTrainer and a PEFT LoRA adapter.

    python scripts/train_lora_dpo.py --config configs/lora_dpo.yaml

This is the one task where TRL and PEFT are allowed. Rows from
``data/persona_lora/ramsay_style_preferences.json`` become TRL conversational
records; the model's own chat template formats them in training and in
generation. With a LoRA adapter, TRL uses the base model with the adapter
disabled as the frozen reference, so no second copy is loaded.

Writes the adapter, a merged fp16 model for generation, ``metrics.jsonl`` in
the same schema as the from-scratch runs, and ``summary.json``.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

from smalign.train_utils import load_config, set_seed

DEFAULTS = {
    "model": "HuggingFaceTB/SmolLM2-360M-Instruct",
    "data_path": "data/persona_lora/ramsay_style_preferences.json",
    "out_dir": "outputs/lora_dpo",
    "n_eval": 20,
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "lr": 5e-5,
    "beta": 0.1,
    "loss_type": ["sigmoid"],   # ["sigmoid", "sft"] adds an NLL term on the chosen response
    "loss_weights": None,
    "batch_size": 4,
    "grad_accum": 2,
    "epochs": 3,
    "warmup_ratio": 0.1,
    "max_length": 256,
    "eval_steps": 5,
    "fp16": True,
    "seed": 42,
    "max_train": None,
}


def to_conversational(row: dict) -> dict:
    return {
        "prompt": [{"role": "user", "content": row["prompt"]}],
        "chosen": [{"role": "assistant", "content": row["chosen"]}],
        "rejected": [{"role": "assistant", "content": row["rejected"]}],
    }


def convert_log(history: list[dict]) -> list[dict]:
    """TRL log_history -> the metrics.jsonl schema used by plot_metrics.py."""

    rows = []
    for h in history:
        step = h.get("step", 0)
        if "loss" in h and "rewards/accuracies" in h:
            rows.append({"kind": "train", "step": step, "train_loss": h["loss"],
                         "train_reward_acc": h["rewards/accuracies"], "train_reward_margin": h["rewards/margins"],
                         "train_chosen_reward": h["rewards/chosen"], "train_rejected_reward": h["rewards/rejected"],
                         "lr": h.get("learning_rate"), "grad_norm": h.get("grad_norm")})
        elif "eval_loss" in h:
            rows.append({"kind": "eval", "step": step, "val_loss": h["eval_loss"],
                         "val_reward_acc": h["eval_rewards/accuracies"], "val_reward_margin": h["eval_rewards/margins"],
                         "val_chosen_reward": h["eval_rewards/chosen"], "val_rejected_reward": h["eval_rewards/rejected"]})
    return rows


def main() -> None:
    cfg = load_config(DEFAULTS)
    set_seed(cfg["seed"])
    out = Path(cfg["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    if (out / "summary.json").exists():
        print(f"{out / 'summary.json'} exists; nothing to do.")
        return

    rows = json.load(open(cfg["data_path"], encoding="utf-8"))
    order = list(range(len(rows)))
    random.Random(cfg["seed"]).shuffle(order)
    eval_idx, train_idx = order[: cfg["n_eval"]], order[cfg["n_eval"] :][: cfg["max_train"]]
    (out / "split.json").write_text(json.dumps({"eval_indices": sorted(eval_idx), "train_indices": sorted(train_idx)}))
    train_ds = Dataset.from_list([to_conversational(rows[i]) for i in train_idx])
    eval_ds = Dataset.from_list([to_conversational(rows[i]) for i in eval_idx])

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(cfg["model"], torch_dtype=torch.float32)

    peft_config = LoraConfig(r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"], lora_dropout=cfg["lora_dropout"],
                             target_modules=cfg["target_modules"], bias="none", task_type="CAUSAL_LM")
    use_fp16 = cfg["fp16"] and torch.cuda.is_available()
    args = DPOConfig(
        output_dir=str(out / "trainer"),
        beta=cfg["beta"],
        loss_type=list(cfg["loss_type"]),
        loss_weights=list(cfg["loss_weights"]) if cfg["loss_weights"] else None,
        learning_rate=cfg["lr"],
        lr_scheduler_type="cosine",
        warmup_ratio=cfg["warmup_ratio"],
        per_device_train_batch_size=cfg["batch_size"],
        per_device_eval_batch_size=cfg["batch_size"],
        gradient_accumulation_steps=cfg["grad_accum"],
        num_train_epochs=cfg["epochs"],
        max_length=cfg["max_length"],
        fp16=use_fp16,
        bf16=False,  # T4 has no fast bf16
        logging_steps=1,
        eval_strategy="steps",
        eval_steps=cfg["eval_steps"],
        save_strategy="no",
        report_to=[],
        seed=cfg["seed"],
    )
    trainer = DPOTrainer(model=model, args=args, train_dataset=train_ds, eval_dataset=eval_ds,
                         processing_class=tokenizer, peft_config=peft_config)
    trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    print(f"trainable LoRA parameters: {trainable:,}")

    first_eval = trainer.evaluate()
    trainer.state.log_history.insert(0, {**first_eval, "step": 0})
    trainer.train()
    final_eval = trainer.evaluate()

    metrics = convert_log(trainer.state.log_history)
    (out / "metrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in metrics))
    trainer.model.save_pretrained(out / "adapter")
    tokenizer.save_pretrained(out / "adapter")

    merged = trainer.model.merge_and_unload().to(torch.float16)
    merged.save_pretrained(out / "merged", safe_serialization=True)
    tokenizer.save_pretrained(out / "merged")

    summary = {"initial_eval": first_eval, "final_eval": final_eval, "train_pairs": len(train_ds),
               "eval_pairs": len(eval_ds), "optimizer_steps": trainer.state.global_step,
               "trainable_params": trainable, "config": cfg}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps({k: summary[k] for k in ("initial_eval", "final_eval", "optimizer_steps")}, indent=2, default=str))


if __name__ == "__main__":
    main()
