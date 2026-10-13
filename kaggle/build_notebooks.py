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
# Let PyTorch grow memory segments instead of fragmenting (inherited by every ! and subprocess call).
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

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


# Persona LoRA variants: (run folder, config, generation file stem). v1 is the first attempt; v2-v4 the sweep.
LORA_VARIANTS = [
    ("lora_dpo", "configs/lora_dpo.yaml", "smollm2_lora_dpo"),
    ("lora_dpo_v2", "configs/lora_dpo_v2.yaml", "smollm2_lora_dpo_v2"),
    ("lora_dpo_v3", "configs/lora_dpo_v3.yaml", "smollm2_lora_dpo_v3"),
    ("lora_dpo_v4", "configs/lora_dpo_v4.yaml", "smollm2_lora_dpo_v4"),
    ("lora_dpo_v5", "configs/lora_dpo_v5.yaml", "smollm2_lora_dpo_v5"),
    ("lora_dpo_v6", "configs/lora_dpo_v6.yaml", "smollm2_lora_dpo_v6"),
]

PERSONA_RUNS = r"""
# Each variant trains (skipped if its summary.json exists), then answers the persona set (finished rows skipped).
# GPU 1 also generates the untuned SmolLM2 baseline. Two chains run in parallel.
VARIANTS = VARIANTS_HERE
gen = ("--eval_set eval_sets/persona_style_eval_40.json --batch_size 20 --max_new_tokens 256 --chat_template")

def variant_chain(run, config, stem):
    return (f"mkdir -p outputs/{run} && python -u scripts/train_lora_dpo.py --config {config} >> outputs/{run}/train.log 2>&1 && "
            f"python -u scripts/generate.py --model outputs/{run}/merged --generator smollm2-360m-{run.replace('_', '-')} "
            f"--out outputs/gen_persona/{stem}.json {gen}")

base_gen = (f"python -u scripts/generate.py --model HuggingFaceTB/SmolLM2-360M-Instruct --generator smollm2-360m-instruct "
            f"--out outputs/gen_persona/smollm2_base.json {gen}")
chains = {"0": [variant_chain(*v) for v in VARIANTS[0::2]], "1": [base_gen] + [variant_chain(*v) for v in VARIANTS[1::2]]}
os.makedirs("outputs/gen_persona", exist_ok=True)
procs = {gpu: subprocess.Popen(" && ".join(cmds), shell=True, env={**os.environ, "CUDA_VISIBLE_DEVICES": gpu, "PYTHONPATH": "."},
                               stdout=open(f"outputs/gen_persona/gpu{gpu}.log", "a"), stderr=subprocess.STDOUT)
         for gpu, cmds in chains.items()}
print("exit codes", {gpu: p.wait() for gpu, p in procs.items()})
!tail -n 4 outputs/gen_persona/gpu0.log outputs/gen_persona/gpu1.log
for run, _, _ in VARIANTS:
    print(f"== {run}"); os.system(f"grep -E 'trainable|Traceback|Error' outputs/{run}/train.log | tail -3")
"""

PERSONA_REPORT = r"""
import json
for run, _, _ in VARIANTS:
    os.system(f"PYTHONPATH=. python scripts/plot_metrics.py outputs/{run}")
    s = json.load(open(f"outputs/{run}/summary.json"))
    f = s["final_eval"]
    print(f"{run:12s} steps={s['optimizer_steps']} eval_acc={f['eval_rewards/accuracies']:.3f} "
          f"margin={f['eval_rewards/margins']:.3f} loss={f['eval_loss']:.3f}")
outs = {stem: {r["id"]: r for r in json.load(open(f"outputs/gen_persona/{stem}.json"))} for stem in
        ["smollm2_base"] + [v[2] for v in VARIANTS]}
for stem, rows in outs.items():
    n_len = sum(r["finish_reason"] == "length" for r in rows.values())
    print(f"{stem:22s} hit_max_tokens={n_len:2d} mean_words={sum(len(r['output'].split()) for r in rows.values()) / len(rows):.0f}")
for rid in list(outs["smollm2_base"])[::8]:
    print("\n###", outs["smollm2_base"][rid]["category"], "|", outs["smollm2_base"][rid]["instruction"])
    for stem, rows in outs.items():
        print(f"  [{stem}]", rows[rid]["output"][:300].replace("\n", " "))
"""


def build_lora() -> None:
    variants = repr([list(v) for v in LORA_VARIANTS])
    runs = [v[0] for v in LORA_VARIANTS]
    cells = [
        md("""
# PA2 stage 04: persona LoRA-DPO with TRL (Task 5)

`HuggingFaceTB/SmolLM2-360M-Instruct` + LoRA trained with TRL `DPOTrainer` on
`data/persona_lora/ramsay_style_preferences.json` (200 train / 20 held out). v1 is the first
configuration (over-optimised); v2-v4 lower the learning rate, add an SFT term on the chosen
response, and raise beta (too weak to change generations); v5-v6 restore update strength with the SFT anchor. Every variant and the untuned model answer the fixed 40-prompt
persona style set with identical greedy decoding.
"""),
        *setup_cells(extra_pip='"peft==0.21.2" "trl==1.15.0"'),
        restore_cell([*runs, "gen_persona"]),
        code(PERSONA_RUNS.replace("VARIANTS_HERE", variants)),
        code(PERSONA_REPORT),
        code(f"""
!cd outputs && zip -qr /kaggle/working/lora_dpo_run_outputs.zip {" ".join(runs)} gen_persona -x "*/merged/*" "*/trainer/*"
!du -sh outputs/lora_dpo* /kaggle/working/lora_dpo_run_outputs.zip
"""),
    ]
    write_stage("04_lora", "pa2-04-lora-dpo", "PA2 04 LoRA DPO", cells)


FIND_SFT = """
# The SFT stage's output is attached as an input; use its best checkpoint as policy init and reference.
hits = sorted(glob.glob("/kaggle/input/**/small-model-alignment-starter/outputs/sft/best/config.json", recursive=True))
assert hits, "attach the pa2-01-sft notebook output"
SFT_BEST = os.path.dirname(hits[-1])
print("SFT best checkpoint:", SFT_BEST)
!cat {SFT_BEST}/checkpoint_info.json
"""


DPO_EVAL_GEN = r"""
# Task 6 on Kaggle: once training has finished, the DPO best and SFT best checkpoints answer this
# run's fixed evaluation set (one model per GPU, identical decoding), then the blind pair file is built.
import json
summary = json.load(open("outputs/RUN/summary.json"))
assert not summary["stopped_early"], "training hit the time limit: push again with --resume to finish it first"
FLAGS = "DECODING_FLAGS"
os.makedirs("outputs/gen_eval", exist_ok=True)
jobs = {"0": ("RUN", "outputs/RUN/best"), "1": ("sft", SFT_BEST)}
procs = [subprocess.Popen(
            f"python -u scripts/generate.py --model {path} --generator qwen2.5-0.5b-{name} "
            f"--eval_set eval_sets/EVALSET.json --out outputs/gen_eval/EVALSET__{name}.json {FLAGS}",
            shell=True, env={**os.environ, "CUDA_VISIBLE_DEVICES": gpu, "PYTHONPATH": "."},
            stdout=open(f"outputs/gen_eval/gen_{name}.log", "a"), stderr=subprocess.STDOUT)
         for gpu, (name, path) in jobs.items()]
print("exit codes", [p.wait() for p in procs])
!tail -n 2 outputs/gen_eval/gen_RUN.log outputs/gen_eval/gen_sft.log
!PYTHONPATH=. python scripts/build_pairs.py --a outputs/gen_eval/EVALSET__sft.json --a_name sft \
    --b outputs/gen_eval/EVALSET__RUN.json --b_name RUN --out outputs/gen_eval/pairs_EVALSET__RUN_vs_sft.jsonl
for f in sorted(glob.glob("outputs/gen_eval/EVALSET__*.json")):
    if f.endswith(".settings.json"):
        continue
    rows = json.load(open(f))
    n_len = sum(r["finish_reason"] == "length" for r in rows)
    print(f"{os.path.basename(f):45s} rows={len(rows)} hit_max_tokens={n_len} mean_words={sum(len(r['output'].split()) for r in rows) / len(rows):.0f}")
"""


def build_dpo(folder: str, slug: str, title: str, run: str, config: str, evalset: str, blurb: str) -> None:
    flags = f"--batch_size 32 --max_new_tokens {DECODING['max_new_tokens']} --repetition_penalty {DECODING['repetition_penalty']}"
    cells = [
        md(f"""
# PA2 stage {folder[:2]}: {title} (Task 4)

{blurb} From-scratch DPO (`scripts/train_dpo.py`): policy on `cuda:0`, frozen reference on
`cuda:1`, both from the SFT `best` checkpoint. Resumes from `outputs/{run}/resume` when a
previous run's output is attached. After training, the DPO and SFT best checkpoints answer
`eval_sets/{evalset}.json` with identical greedy decoding (`{flags}`) and the blind pair file
for DPO-vs-SFT judging is written to `outputs/gen_eval/`.
"""),
        *setup_cells(),
        code(FIND_SFT),
        restore_cell([run, "gen_eval"]),
        code(f"""
!mkdir -p outputs/{run}
!PYTHONPATH=. python -u scripts/train_dpo.py --config {config} --sft_checkpoint {{SFT_BEST}} 2>&1 | tee -a outputs/{run}/train.log
"""),
        code(f"""
!PYTHONPATH=. python scripts/plot_metrics.py outputs/{run}
!cat outputs/{run}/summary.json
"""),
        code(DPO_EVAL_GEN.replace("RUN", run).replace("EVALSET", evalset).replace("DECODING_FLAGS", flags)),
        code(f"""
!cd outputs && zip -qr /kaggle/working/{run}_run_outputs.zip {run} gen_eval -x "{run}/best/*" "{run}/final/*" "{run}/resume/*"
!du -sh outputs/{run}/* /kaggle/working/{run}_run_outputs.zip
"""),
    ]
    write_stage(folder, slug, f"PA2 {folder[:2]} {title}", cells, kernel_sources=[f"{OWNER}/pa2-01-sft"])


# One decoding policy for every model in a comparison (set from the stage-01 baseline evidence).
DECODING = {"max_new_tokens": 512, "repetition_penalty": 1.1}

FIND_EVAL_CKPTS = """
import json

def find_ckpt(run):
    hits = sorted(glob.glob(f"/kaggle/input/**/small-model-alignment-starter/outputs/{run}/best/config.json", recursive=True))
    assert hits, f"attach the stage output that contains outputs/{run}/best"
    return os.path.dirname(hits[-1])

CKPT = {"sft": find_ckpt("sft"), "dpo_hh": find_ckpt("dpo_hh"), "dpo_uf": find_ckpt("dpo_uf")}
print(json.dumps(CKPT, indent=2))
"""

EVAL_GENERATION = """
# GPU 0: SFT on both sets. GPU 1: each DPO model on its own set. Two processes in parallel.
jobs = {
    "0": [("sft", "alpaca_eval_300_seed42"), ("sft", "hh_rlhf_eval_300_seed42")],
    "1": [("dpo_uf", "alpaca_eval_300_seed42"), ("dpo_hh", "hh_rlhf_eval_300_seed42")],
}
os.makedirs("outputs/gen_eval", exist_ok=True)
procs = []
for gpu, runs in jobs.items():
    cmd = " && ".join(
        f"python -u scripts/generate.py --model {CKPT[model]} --generator qwen2.5-0.5b-{model} "
        f"--eval_set eval_sets/{evalset}.json --out outputs/gen_eval/{evalset}__{model}.json FLAGS"
        for model, evalset in runs)
    procs.append(subprocess.Popen(cmd, shell=True, env={**os.environ, "CUDA_VISIBLE_DEVICES": gpu, "PYTHONPATH": "."},
                                  stdout=open(f"outputs/gen_eval/gpu{gpu}.log", "a"), stderr=subprocess.STDOUT))
print("exit codes", [p.wait() for p in procs])
!tail -n 3 outputs/gen_eval/gpu0.log outputs/gen_eval/gpu1.log
"""

BUILD_EVAL_PAIRS = """
!PYTHONPATH=. python scripts/build_pairs.py --a outputs/gen_eval/alpaca_eval_300_seed42__sft.json --a_name sft \\
    --b outputs/gen_eval/alpaca_eval_300_seed42__dpo_uf.json --b_name dpo_uf --out outputs/gen_eval/pairs_alpaca_dpo_uf_vs_sft.jsonl
!PYTHONPATH=. python scripts/build_pairs.py --a outputs/gen_eval/hh_rlhf_eval_300_seed42__sft.json --a_name sft \\
    --b outputs/gen_eval/hh_rlhf_eval_300_seed42__dpo_hh.json --b_name dpo_hh --out outputs/gen_eval/pairs_hh_dpo_hh_vs_sft.jsonl
for f in sorted(glob.glob("outputs/gen_eval/*__*.json")):
    if f.endswith(".settings.json"):
        continue
    rows = json.load(open(f))
    n_len = sum(r["finish_reason"] == "length" for r in rows)
    mean_words = sum(len(r["output"].split()) for r in rows) / len(rows)
    print(f"{os.path.basename(f):45s} rows={len(rows)} hit_max_tokens={n_len} mean_words={mean_words:.0f}")
"""


def build_generate() -> None:
    flags = f"--batch_size 32 --max_new_tokens {DECODING['max_new_tokens']} --repetition_penalty {DECODING['repetition_penalty']}"
    cells = [
        md(f"""
# PA2 stage 05: evaluation generations and blind pair files (Task 6)

SFT vs DPO-UltraFeedback on the fixed AlpacaEval subset (300 prompts, seed 42), and SFT vs
DPO-HH on the provided helpful/harmless set (300 prompts). Same prompt format and decoding for
every model: greedy, `{flags}`. Generation resumes per row; pair files are written before any judging.
"""),
        *setup_cells(),
        code(FIND_EVAL_CKPTS),
        restore_cell(["gen_eval"]),
        code(EVAL_GENERATION.replace("FLAGS", flags)),
        code(BUILD_EVAL_PAIRS),
        code("""
!cd outputs && zip -qr /kaggle/working/gen_eval_outputs.zip gen_eval
!du -sh outputs/gen_eval /kaggle/working/gen_eval_outputs.zip
"""),
    ]
    write_stage("05_generate", "pa2-05-generate", "PA2 05 Generate", cells,
                kernel_sources=[f"{OWNER}/pa2-01-sft", f"{OWNER}/pa2-02-dpo-hh-rlhf", f"{OWNER}/pa2-03-dpo-ultrafeedback"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", action="append", default=[], help="stage folder(s) to rebuild with their own output attached")
    RESUME.update(ap.parse_args().resume)
    build_sft()
    build_dpo("02_dpo_hh", "pa2-02-dpo-hh-rlhf", "DPO HH-RLHF", "dpo_hh", "configs/dpo_hh.yaml", "hh_rlhf_eval_300_seed42",
              "15k single-turn Anthropic HH-RLHF pairs (helpful/harmless), validated on 1k held-out pairs.")
    build_dpo("03_dpo_uf", "pa2-03-dpo-ultrafeedback", "DPO UltraFeedback", "dpo_uf", "configs/dpo_uf.yaml", "alpaca_eval_300_seed42",
              "15k UltraFeedback-binarized pairs (general helpfulness), validated on 1k held-out pairs.")
    build_lora()
    build_generate()
