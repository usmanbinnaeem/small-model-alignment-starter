"""Build the fixed SFT, DPO, and AlpacaEval subsets.

Writes JSONL splits to ``--out_dir`` (uploaded to Kaggle as a dataset) and the
AlpacaEval subset to ``eval_sets/``. A manifest records sources, filters, and
counts so the subsets can be audited and rebuilt.

    python scripts/prepare_data.py --out_dir data/prepared
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer

from smalign import data

MODEL = "Qwen/Qwen2.5-0.5B"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="data/prepared")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--max_prompt_length", type=int, default=256)
    ap.add_argument("--sft_train", type=int, default=30_000)
    ap.add_argument("--sft_val", type=int, default=1_000)
    ap.add_argument("--dpo_train", type=int, default=15_000)
    ap.add_argument("--dpo_val", type=int, default=1_000)
    args = ap.parse_args()

    out = Path(args.out_dir)
    tok = AutoTokenizer.from_pretrained(MODEL)
    manifest: dict = {"tokenizer": MODEL, "seed": args.seed, "max_length": args.max_length,
                      "max_prompt_length": args.max_prompt_length, "splits": {}}

    def record(name, source, n_raw, n_filtered, rows, filters):
        data.write_jsonl(out / f"{name}.jsonl", rows)
        manifest["splits"][name] = {"source": source, "rows_after_single_turn": n_raw,
                                    "rows_after_filters": n_filtered, "sampled": len(rows), "filters": filters}
        print(f"{name:12s} {source:50s} single-turn={n_raw:7d} filtered={n_filtered:7d} sampled={len(rows)}")

    # SFT: UltraChat, first turn only, must fit in max_length with EOS.
    uc = load_dataset("HuggingFaceH4/ultrachat_200k")
    sft_filter = f"formatted prompt <= {args.max_prompt_length} tokens and prompt + response + EOS <= {args.max_length}"
    for name, split, n in (("sft_train", "train_sft", args.sft_train), ("sft_val", "test_sft", args.sft_val)):
        rows = data.ultrachat_rows(uc[split])
        kept = data.keep_sft_fitting(rows, tok, args.max_length, args.max_prompt_length)
        record(name, f"HuggingFaceH4/ultrachat_200k:{split}", len(rows), len(kept), data.sample(kept, n, args.seed), sft_filter)

    dpo_filter = f"formatted prompt <= {args.max_prompt_length} tokens; responses truncated at {args.max_length} total"

    hh = load_dataset("Anthropic/hh-rlhf")
    for name, split, n in (("hh_train", "train", args.dpo_train), ("hh_val", "test", args.dpo_val)):
        rows = data.hh_rows(hh[split])
        kept = data.keep_short_prompts(rows, tok, args.max_prompt_length)
        record(name, f"Anthropic/hh-rlhf:{split}", len(rows), len(kept), data.sample(kept, n, args.seed),
               "single-turn, identical prompt, chosen != rejected; " + dpo_filter)

    uf = load_dataset("HuggingFaceH4/ultrafeedback_binarized")
    for name, split, n in (("uf_train", "train_prefs", args.dpo_train), ("uf_val", "test_prefs", args.dpo_val)):
        rows = data.ultrafeedback_rows(uf[split])
        kept = data.keep_short_prompts(rows, tok, args.max_prompt_length)
        record(name, f"HuggingFaceH4/ultrafeedback_binarized:{split}", len(rows), len(kept),
               data.sample(kept, n, args.seed), "score_chosen > score_rejected, chosen != rejected; " + dpo_filter)

    alpaca = data.alpaca_eval_subset(300, args.seed)
    Path("eval_sets").mkdir(exist_ok=True)
    with open("eval_sets/alpaca_eval_300_seed42.json", "w", encoding="utf-8") as f:
        json.dump(alpaca, f, ensure_ascii=False, indent=1)
    manifest["alpaca_eval"] = {"source": "tatsu-lab/alpaca_eval:alpaca_eval.json",
                               "rule": "sorted(random.Random(42).sample(range(805), 300))", "n": len(alpaca)}

    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    Path("data/prepared_manifest.json").write_text(json.dumps(manifest, indent=2))
    print("wrote", out, "and eval_sets/alpaca_eval_300_seed42.json")


if __name__ == "__main__":
    main()
