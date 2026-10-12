"""Write the Kaggle notebook for each pipeline stage, sharing one set of setup cells.

    python kaggle/build_notebooks.py

Each stage folder gets a notebook and a ``kernel-metadata.json``; push one with
``kaggle kernels push -p kaggle/<stage>``. Stages attach the prepared-data
dataset and, where needed, earlier stages' outputs as inputs.

Resuming after a failed or timed-out run:

    python kaggle/build_notebooks.py --resume 01_sft
    kaggle kernels push -p kaggle/01_sft

``--resume`` attaches the stage's own previous output. The notebook copies its
run folders back from that input, training continues from the last saved step,
finished runs are skipped, and generation skips rows already written.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

OWNER = "usmanbinnaeem"
REPO_URL = "https://github.com/usmanbinnaeem/small-model-alignment-starter.git"
DATA_DATASET = f"{OWNER}/pa2-alignment-data"
HERE = Path(__file__).parent


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(keepends=True)}


def restore_cell(run_dirs: list[str]) -> dict:
    return code(f'''
# Resume support: copy this stage's run folders back from a previous run's attached output.
import shutil
for run in {run_dirs!r}:
    local = os.path.join("outputs", run)
    if os.path.exists(local):
        continue
    found = sorted(glob.glob(f"/kaggle/input/**/small-model-alignment-starter/outputs/{{run}}", recursive=True))
    if found:
        print(f"restoring outputs/{{run}} from {{found[-1]}}")
        shutil.copytree(found[-1], local, symlinks=False)
    else:
        print(f"outputs/{{run}}: starting fresh")
!du -sh outputs/* 2>/dev/null || true
''')


def setup_cells(branch: str = "main", extra_pip: str = "") -> list[dict]:
    return [
        code(f'''
REPO_URL = "{REPO_URL}"
BRANCH = "{branch}"
REPO_DIR = "/kaggle/working/small-model-alignment-starter"
'''),
        code("!nvidia-smi"),
        code('''
import os, subprocess
if not os.path.isdir(REPO_DIR):
    subprocess.run(["git", "clone", "--depth", "1", "-b", BRANCH, REPO_URL, REPO_DIR], check=True)
os.chdir(REPO_DIR)
!git log -1 --format="%h %s"
'''),
        code(f'''
!python -m pip uninstall -y -q torchao || true
!python -m pip install -q "transformers>=4.43.0,<5" "datasets>=2.18,<5" huggingface_hub accelerate tqdm pyyaml sentencepiece safetensors pytest matplotlib {extra_pip}
!python -m pip install -q -e . --no-deps
'''),
        code('''
import sys, torch, transformers, datasets
print("python      ", sys.version.split()[0])
print("torch       ", torch.__version__, "| cuda", torch.version.cuda)
print("transformers", transformers.__version__)
print("datasets    ", datasets.__version__)
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"cuda:{i}  {p.name}  {p.total_memory / 2**30:.1f} GiB")
assert torch.cuda.device_count() == 2, "Expected GPU T4 x2"
'''),
        code('''
# Link the attached prepared-data dataset to data/prepared, wherever Kaggle mounted it.
import glob
hits = glob.glob("/kaggle/input/**/sft_train.jsonl", recursive=True)
assert hits, "attach the pa2-alignment-data dataset"
DATA_DIR = os.path.dirname(hits[0])
os.makedirs("data", exist_ok=True)
if not os.path.exists("data/prepared"):
    os.symlink(DATA_DIR, "data/prepared")
!ls -la data/prepared/
!PYTHONPATH=. python -m pytest -q tests/
'''),
    ]


def metadata(slug: str, title: str, code_file: str, kernel_sources: list[str] | None = None) -> dict:
    return {
        "id": f"{OWNER}/{slug}",
        "title": title,
        "code_file": code_file,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "machine_shape": "NvidiaTeslaT4",
        "enable_internet": True,
        "dataset_sources": [DATA_DATASET],
        "competition_sources": [],
        "kernel_sources": kernel_sources or [],
        "model_sources": [],
    }


RESUME: set[str] = set()


def write_stage(folder: str, slug: str, title: str, cells: list[dict], kernel_sources: list[str] | None = None) -> None:
    if folder in RESUME:
        kernel_sources = [*(kernel_sources or []), f"{OWNER}/{slug}"]
    out = HERE / folder
    out.mkdir(parents=True, exist_ok=True)
    nb = {
        "cells": cells,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "language_info": {"name": "python"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    (out / f"{folder.split('_', 1)[1]}.ipynb").write_text(json.dumps(nb, indent=1))
    meta = metadata(slug, title, f"{folder.split('_', 1)[1]}.ipynb", kernel_sources)
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    print("wrote", out)


BASELINE_RUNS = r'''
# Task 2 baseline on the idle second GPU while SFT trains on the first.
# Two decoding settings, so the shared SFT/DPO setting can be chosen with evidence.
baseline_cmd = " && ".join(
    f"python -u scripts/generate.py --model Qwen/Qwen2.5-0.5B --generator qwen2.5-0.5b-base "
    f"--eval_set eval_sets/{evalset}.json --out outputs/gen/baseline/{tag}/{evalset}.json "
    f"--batch_size 32 --max_new_tokens 512 --repetition_penalty {rp}"
    for tag, rp in (("greedy", 1.0), ("greedy_rp1.1", 1.1))
    for evalset in ("alpaca_eval_300_seed42", "hh_rlhf_eval_300_seed42")
)
os.makedirs("outputs/gen/baseline", exist_ok=True)
baseline = subprocess.Popen(baseline_cmd, shell=True, env={**os.environ, "CUDA_VISIBLE_DEVICES": "1", "PYTHONPATH": "."},
                            stdout=open("outputs/gen/baseline/generate.log", "w"), stderr=subprocess.STDOUT)
print("baseline generation started, pid", baseline.pid)
'''


def build_sft() -> None:
    cells = [
        md("""
# PA2 stage 01: SFT (Task 3) and base-model baseline (Task 2)

Trains Qwen2.5-0.5B on 30k UltraChat examples on `cuda:0` with the from-scratch loop in
`scripts/train_sft.py`, while `cuda:1` generates base-model baselines. Outputs land in
`/kaggle/working/small-model-alignment-starter/outputs/`.
"""),
        *setup_cells(),
        restore_cell(["sft", "gen"]),
        code(BASELINE_RUNS),
        code('''
!mkdir -p outputs/sft
!CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python -u scripts/train_sft.py --config configs/sft.yaml 2>&1 | tee -a outputs/sft/train.log
'''),
        code('''
print("waiting for baseline generation ...")
print("exit code", baseline.wait())
!tail -n 5 outputs/gen/baseline/generate.log
'''),
        code('''
!PYTHONPATH=. python scripts/plot_metrics.py outputs/sft
!cat outputs/sft/summary.json
'''),
        code('''
# Task 6 sanity check: a few prompts through the SFT best checkpoint.
!PYTHONPATH=. python scripts/generate.py --model outputs/sft/best --generator sft-sanity \\
    --eval_set eval_sets/alpaca_eval_300_seed42.json --out outputs/gen/sanity/sft_best.json --limit 5 --batch_size 5
import json
for r in json.load(open("outputs/gen/sanity/sft_best.json")):
    print("###", r["instruction"][:200], f"[{r['finish_reason']}]")
    print(r["output"][:800], "\\n")
'''),
        code('''
# Small artifacts (no checkpoints) as one zip, per the assignment's saving guidance.
!cd outputs && zip -qr ../sft_run_outputs.zip sft gen -x "sft/best/*" "sft/final/*" && cd .. && mv sft_run_outputs.zip /kaggle/working/
!du -sh outputs/sft/best outputs/sft/final /kaggle/working/sft_run_outputs.zip
'''),
    ]
    write_stage("01_sft", "pa2-01-sft", "PA2 01 SFT", cells)


PERSONA_GEN = r"""
# Persona eval set through the untuned SmolLM2 (cuda:1) and the LoRA-DPO merge (cuda:0), in parallel.
common = ("--eval_set eval_sets/persona_style_eval_40.json --batch_size 20 --max_new_tokens 256 --chat_template")
jobs = [
    ("1", f"python -u scripts/generate.py --model HuggingFaceTB/SmolLM2-360M-Instruct --generator smollm2-360m-instruct "
          f"--out outputs/gen_persona/smollm2_base.json {common}"),
    ("0", f"python -u scripts/generate.py --model outputs/lora_dpo/merged --generator smollm2-360m-lora-dpo "
          f"--out outputs/gen_persona/smollm2_lora_dpo.json {common}"),
]
os.makedirs("outputs/gen_persona", exist_ok=True)
procs = [subprocess.Popen(cmd, shell=True, env={**os.environ, "CUDA_VISIBLE_DEVICES": gpu, "PYTHONPATH": "."})
         for gpu, cmd in jobs]
print("exit codes", [p.wait() for p in procs])
"""


def build_lora() -> None:
    cells = [
        md("""
# PA2 stage 04: persona LoRA-DPO with TRL (Task 5)

`HuggingFaceTB/SmolLM2-360M-Instruct` + LoRA trained with TRL `DPOTrainer` on
`data/persona_lora/ramsay_style_preferences.json` (200 train / 20 held out). Then the
untuned and LoRA-DPO models answer the fixed 40-prompt persona style set.
"""),
        *setup_cells(extra_pip='"peft==0.21.2" "trl==1.15.0"'),
        restore_cell(["lora_dpo", "gen_persona"]),
        code("""
!mkdir -p outputs/lora_dpo
!CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python -u scripts/train_lora_dpo.py --config configs/lora_dpo.yaml 2>&1 | tee -a outputs/lora_dpo/train.log
"""),
        code(PERSONA_GEN),
        code("""
!PYTHONPATH=. python scripts/plot_metrics.py outputs/lora_dpo
import json
base = {r["id"]: r for r in json.load(open("outputs/gen_persona/smollm2_base.json"))}
lora = {r["id"]: r for r in json.load(open("outputs/gen_persona/smollm2_lora_dpo.json"))}
for rid in list(base)[::5]:
    print("###", base[rid]["category"], "|", base[rid]["instruction"])
    print("BASE:", base[rid]["output"][:400])
    print("LORA:", lora[rid]["output"][:400], "\\n")
"""),
        code("""
!cd outputs && zip -qr /kaggle/working/lora_dpo_run_outputs.zip lora_dpo gen_persona -x "lora_dpo/merged/*" "lora_dpo/trainer/*"
!du -sh outputs/lora_dpo/* /kaggle/working/lora_dpo_run_outputs.zip
"""),
    ]
    write_stage("04_lora", "pa2-04-lora-dpo", "PA2 04 LoRA DPO", cells)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", action="append", default=[], help="stage folder(s) to rebuild with their own output attached")
    RESUME.update(ap.parse_args().resume)
    build_sft()
    build_lora()
