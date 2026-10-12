"""Build a blind pairwise comparison file from two generation files.

    python scripts/build_pairs.py --a outputs/gen/alpaca_sft.json --a_name sft \
        --b outputs/gen/alpaca_dpo_uf.json --b_name dpo_uf --out outputs/eval/alpaca_dpo_uf_vs_sft_pairs.jsonl

Each row shows the judge only ``instruction``, ``response_a``, and
``response_b``. Which model produced which side is randomised per pair (seeded)
and stored under ``hidden``, which the judge never reads.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

PROMPT_FIELDS = ("category", "expected_behavior")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="generation file of the baseline model")
    ap.add_argument("--a_name", required=True)
    ap.add_argument("--b", required=True, help="generation file of the model being compared")
    ap.add_argument("--b_name", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    a = {r["id"]: r for r in json.load(open(args.a, encoding="utf-8"))}
    b = {r["id"]: r for r in json.load(open(args.b, encoding="utf-8"))}
    if a.keys() != b.keys():
        raise SystemExit(f"prompt sets differ: {len(a.keys() ^ b.keys())} ids in only one file")

    rng = random.Random(args.seed)
    rows = []
    for pid, ra in a.items():
        rb = b[pid]
        if ra["instruction"] != rb["instruction"]:
            raise SystemExit(f"{pid}: instructions differ")
        swap = rng.random() < 0.5
        first, second = ((args.b_name, rb), (args.a_name, ra)) if swap else ((args.a_name, ra), (args.b_name, rb))
        rows.append({
            "pair_id": pid,
            "instruction": ra["instruction"],
            "response_a": first[1]["output"],
            "response_b": second[1]["output"],
            **{k: ra[k] for k in PROMPT_FIELDS if k in ra},
            "hidden": {"A": first[0], "B": second[0], "generator_A": first[1]["generator"],
                       "generator_B": second[1]["generator"], "baseline": args.a_name, "candidate": args.b_name},
        })

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    n_swapped = sum(r["hidden"]["A"] == args.b_name for r in rows)
    print(f"wrote {len(rows)} pairs to {out} ({n_swapped} with {args.b_name} as Response A)")


if __name__ == "__main__":
    main()
