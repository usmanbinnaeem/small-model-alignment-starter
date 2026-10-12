"""Pieces shared by the SFT and DPO training loops."""

from __future__ import annotations

import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch


def _coerce(value):
    # PyYAML reads "2e-5" as a string (YAML 1.1 wants a dot), so retry numbers by hand.
    if isinstance(value, str):
        for kind in (int, float):
            try:
                return kind(value)
            except ValueError:
                pass
    return value


def load_config(defaults: dict) -> dict:
    """Defaults, then ``--config file.yaml``, then ``--key value`` overrides (parsed as YAML)."""

    import argparse

    import yaml

    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    for key in defaults:
        ap.add_argument(f"--{key}", default=None)
    overrides = vars(ap.parse_args())
    cfg = dict(defaults)
    path = overrides.pop("config")
    if path:
        cfg.update(yaml.safe_load(open(path)) or {})
        cfg["config_file"] = path
    for key, raw in overrides.items():
        if raw is not None:
            cfg[key] = yaml.safe_load(raw)
    return {key: _coerce(value) for key, value in cfg.items()}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_device(index: int = 0) -> torch.device:
    if torch.cuda.is_available():
        return torch.device(f"cuda:{min(index, torch.cuda.device_count() - 1)}")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def decay_groups(model, weight_decay: float) -> list[dict]:
    """Weight decay on matrices only; biases and norm weights are not decayed."""

    decay, no_decay = [], []
    for p in model.parameters():
        if p.requires_grad:
            (decay if p.ndim >= 2 else no_decay).append(p)
    return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]


def warmup_cosine(optimizer, total_steps: int, warmup_ratio: float):
    warmup = max(1, int(round(total_steps * warmup_ratio))) if warmup_ratio > 0 else 0

    def factor(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


class MetricsLog:
    """Append-only JSONL log, flushed on every write so partial runs survive a disconnect."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.start = time.time()

    def write(self, **record) -> None:
        record = {"time_s": round(time.time() - self.start, 1), **record}
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
            f.flush()
            os.fsync(f.fileno())


def save_checkpoint(model, tokenizer, path: str | Path, extra: dict | None = None) -> None:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path, safe_serialization=True)
    tokenizer.save_pretrained(path)
    if extra is not None:
        (path / "checkpoint_info.json").write_text(json.dumps(extra, indent=2))


def autocast(device: torch.device, enabled: bool):
    return torch.autocast(device_type=device.type, dtype=torch.float16, enabled=enabled and device.type == "cuda")


def make_scaler(device: torch.device, enabled: bool):
    return torch.amp.GradScaler("cuda", enabled=enabled and device.type == "cuda")
