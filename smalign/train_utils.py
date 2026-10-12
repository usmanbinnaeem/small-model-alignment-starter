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
    if os.environ.get("SMALIGN_DEVICE"):  # e.g. "cpu" for deterministic local tests
        return torch.device(os.environ["SMALIGN_DEVICE"])
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


RESUME_DIR = "resume"
_STATE_FILE = "state.json"  # written last, so its presence marks a complete save


def save_resume(out_dir: str | Path, model, optimizer, scheduler, scaler, state: dict) -> None:
    """Save everything needed to continue training, replacing the previous save only once complete."""

    out_dir = Path(out_dir)
    final, tmp = out_dir / RESUME_DIR, out_dir / f"{RESUME_DIR}.tmp"
    if tmp.exists():
        _rmtree(tmp)
    tmp.mkdir(parents=True)
    model.save_pretrained(tmp / "model", safe_serialization=True)
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "scaler": scaler.state_dict(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
        tmp / "trainer_state.pt",
    )
    (tmp / _STATE_FILE).write_text(json.dumps(state, indent=2))
    if final.exists():
        _rmtree(final)
    tmp.rename(final)


def find_resume(out_dir: str | Path) -> Path | None:
    """The newest complete resume save under ``out_dir``, if any."""

    for name in (RESUME_DIR, f"{RESUME_DIR}.tmp"):
        path = Path(out_dir) / name
        if (path / _STATE_FILE).exists():
            return path
    return None


def load_resume(path: Path, optimizer, scheduler, scaler) -> dict:
    blob = torch.load(path / "trainer_state.pt", map_location="cpu", weights_only=False)
    optimizer.load_state_dict(blob["optimizer"])
    if scheduler is not None and blob["scheduler"] is not None:
        scheduler.load_state_dict(blob["scheduler"])
    scaler.load_state_dict(blob["scaler"])
    torch.set_rng_state(blob["torch_rng"])
    if blob["cuda_rng"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(blob["cuda_rng"])
    return json.loads((path / _STATE_FILE).read_text())


def drop_resume(out_dir: str | Path) -> None:
    for name in (RESUME_DIR, f"{RESUME_DIR}.tmp"):
        if (Path(out_dir) / name).exists():
            _rmtree(Path(out_dir) / name)


def trim_metrics(path: str | Path, last_step: int) -> None:
    """Drop log records written after ``last_step`` so a resumed run does not duplicate them."""

    path = Path(path)
    if not path.exists():
        return
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    kept = [r for r in rows if r.get("step", 0) <= last_step]
    path.write_text("".join(json.dumps(r) + "\n" for r in kept))


def _rmtree(path: Path) -> None:
    import shutil

    shutil.rmtree(path)


def batch_order(n: int, batch_size: int, seed: int, epoch: int) -> list[list[int]]:
    """Deterministic shuffled batches for an epoch, so a resumed run can skip exactly the ones already seen."""

    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed + epoch)).tolist()
    return [perm[i : i + batch_size] for i in range(0, n, batch_size)]


def autocast(device: torch.device, enabled: bool):
    return torch.autocast(device_type=device.type, dtype=torch.float16, enabled=enabled and device.type == "cuda")


def make_scaler(device: torch.device, enabled: bool):
    return torch.amp.GradScaler("cuda", enabled=enabled and device.type == "cuda")
