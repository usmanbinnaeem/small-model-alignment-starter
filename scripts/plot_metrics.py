"""Plot training curves from one or more runs' ``metrics.jsonl``.

    python scripts/plot_metrics.py outputs/sft
    python scripts/plot_metrics.py outputs/dpo_hh outputs/dpo_uf --out figures/dpo.png

SFT runs get train and validation loss. DPO runs get loss, reward accuracy,
and reward margin. Training curves are drawn raw (light) and with an
exponential moving average (dark); the best checkpoint is marked.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def load(run: Path) -> tuple[list[dict], list[dict]]:
    rows = [json.loads(l) for l in open(run / "metrics.jsonl") if l.strip()]
    return [r for r in rows if r["kind"] == "train"], [r for r in rows if r["kind"] == "eval"]


def ema(values: list[float], alpha: float) -> list[float]:
    out, acc = [], None
    for v in values:
        acc = v if acc is None else alpha * acc + (1 - alpha) * v
        out.append(acc)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--smooth", type=float, default=0.8, help="EMA weight on the previous value")
    args = ap.parse_args()

    is_dpo = "train_reward_acc" in open(args.runs[0] / "metrics.jsonl").read()
    panels = (
        [("loss", "train_loss", "val_loss", "DPO loss"),
         ("acc", "train_reward_acc", "val_reward_acc", "reward accuracy (r_w > r_l)"),
         ("margin", "train_reward_margin", "val_reward_margin", "reward margin (r_w - r_l)")]
        if is_dpo else [("loss", "train_loss", "val_loss", "loss (mean response-token NLL)")]
    )
    fig, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 3.8), squeeze=False)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for run_i, run in enumerate(args.runs):
        train, evals = load(run)
        color = colors[run_i % len(colors)]
        name = run.name
        for ax, (_, train_key, val_key, ylabel) in zip(axes[0], panels):
            steps = [r["step"] for r in train]
            values = [r[train_key] for r in train]
            ax.plot(steps, values, color=color, alpha=0.25, linewidth=1)
            ax.plot(steps, ema(values, args.smooth), color=color, linewidth=1.6,
                    label=f"{name} train (EMA {args.smooth})")
            ax.plot([r["step"] for r in evals], [r[val_key] for r in evals], color=color, linestyle="--",
                    marker="o", markersize=3, linewidth=1.2, label=f"{name} validation")
            best = [r for r in evals if r.get("is_best")]
            if best:
                b = best[-1]
                ax.scatter([b["step"]], [b[val_key]], s=70, facecolors="none", edgecolors=color, linewidths=1.6, zorder=5)
            ax.set_xlabel("optimizer step")
            ax.set_ylabel(ylabel)
            ax.grid(alpha=0.3)

    axes[0][0].legend(fontsize=7)
    fig.suptitle(("DPO" if is_dpo else "SFT") + " training (circle = selected best checkpoint)", fontsize=10)
    fig.tight_layout()
    out = args.out or args.runs[0] / "training_curves.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    print("wrote", out)


if __name__ == "__main__":
    main()
