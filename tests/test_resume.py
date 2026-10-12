"""Interrupted-and-resumed training must match uninterrupted training exactly.

Uses a tiny randomly initialised Qwen2 model with the real Qwen2.5 tokenizer
(downloaded once and cached; skipped when offline). A near-zero time limit
makes each run stop and save after one optimizer step, so the interrupted run
is resumed at every step.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def tiny_checkpoint(tmp_path_factory):
    transformers = pytest.importorskip("transformers")
    try:
        tok = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    except Exception as exc:  # offline
        pytest.skip(f"Qwen tokenizer unavailable: {exc}")
    torch.manual_seed(0)
    config = transformers.Qwen2Config(vocab_size=len(tok), hidden_size=16, intermediate_size=32,
                                      num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
                                      max_position_embeddings=128, tie_word_embeddings=True)
    path = tmp_path_factory.mktemp("tiny") / "model"
    transformers.Qwen2ForCausalLM(config).save_pretrained(path)
    tok.save_pretrained(path)
    return path


def write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def run(script: str, args: dict) -> None:
    cmd = [sys.executable, str(ROOT / "scripts" / script)]
    for key, value in args.items():
        cmd += [f"--{key}", str(value)]
    env = {**os.environ, "PYTHONPATH": str(ROOT), "SMALIGN_DEVICE": "cpu"}
    subprocess.run(cmd, check=True, env=env, cwd=ROOT, capture_output=True, text=True)


def run_until_done(script: str, args: dict, max_runs: int = 50) -> int:
    for n in range(1, max_runs + 1):
        run(script, args)
        if not json.loads((Path(args["out_dir"]) / "summary.json").read_text())["stopped_early"]:
            return n
    raise AssertionError("never finished")


def assert_same_weights(a: Path, b: Path) -> None:
    wa, wb = load_file(a / "model.safetensors"), load_file(b / "model.safetensors")
    assert wa.keys() == wb.keys()
    for key in wa:
        assert torch.equal(wa[key], wb[key]), key


def train_records(run_dir: Path) -> list[dict]:
    rows = [json.loads(l) for l in (run_dir / "metrics.jsonl").read_text().splitlines()]
    return [{k: v for k, v in r.items() if k not in ("time_s", "tokens_per_s")} for r in rows if r["kind"] != "resume"]


def test_sft_resume_matches_uninterrupted(tiny_checkpoint, tmp_path):
    rows = [{"id": str(i), "instruction": f"Say the number {i}.", "response": f"The number is {i}." * (1 + i % 3)}
            for i in range(24)]
    common = {"model": tiny_checkpoint, "train_path": write_jsonl(tmp_path / "train.jsonl", rows),
              "val_path": write_jsonl(tmp_path / "val.jsonl", rows[:4]), "batch_size": 2, "grad_accum": 3,
              "epochs": 2, "lr": 1e-3, "eval_every": 3, "log_every": 1, "fp16": "false"}

    run("train_sft.py", {**common, "out_dir": tmp_path / "straight"})
    runs = run_until_done("train_sft.py", {**common, "out_dir": tmp_path / "resumed", "time_limit_hours": 1e-9})

    assert runs == 8  # 24 rows / (2 x 3) = 4 steps per epoch, 2 epochs, one step per run
    assert_same_weights(tmp_path / "straight" / "final", tmp_path / "resumed" / "final")
    assert_same_weights(tmp_path / "straight" / "best", tmp_path / "resumed" / "best")
    assert train_records(tmp_path / "straight") == train_records(tmp_path / "resumed")
    assert not (tmp_path / "resumed" / "resume").exists()


def test_dpo_resume_matches_uninterrupted(tiny_checkpoint, tmp_path):
    rows = [{"id": str(i), "instruction": f"Pick a word {i}.", "chosen": f"Good answer {i}.", "rejected": f"bad {i} " * 3}
            for i in range(12)]
    common = {"sft_checkpoint": tiny_checkpoint, "train_path": write_jsonl(tmp_path / "train.jsonl", rows),
              "val_path": write_jsonl(tmp_path / "val.jsonl", rows[:4]), "grad_accum": 3, "lr": 1e-3,
              "eval_every": 2, "log_every": 1, "fp16": "false", "beta": 0.5}

    run("train_dpo.py", {**common, "out_dir": tmp_path / "straight"})
    runs = run_until_done("train_dpo.py", {**common, "out_dir": tmp_path / "resumed", "time_limit_hours": 1e-9})

    assert runs == 4
    assert_same_weights(tmp_path / "straight" / "final", tmp_path / "resumed" / "final")
    assert train_records(tmp_path / "straight") == train_records(tmp_path / "resumed")


def test_finished_run_is_not_retrained(tiny_checkpoint, tmp_path):
    rows = [{"id": "0", "instruction": "Hi", "response": "Hello."}] * 4
    args = {"model": tiny_checkpoint, "train_path": write_jsonl(tmp_path / "t.jsonl", rows),
            "val_path": write_jsonl(tmp_path / "v.jsonl", rows), "out_dir": tmp_path / "run",
            "grad_accum": 1, "fp16": "false"}
    run("train_sft.py", args)
    stamp = (tmp_path / "run" / "final" / "model.safetensors").stat().st_mtime_ns
    run("train_sft.py", args)
    assert (tmp_path / "run" / "final" / "model.safetensors").stat().st_mtime_ns == stamp
