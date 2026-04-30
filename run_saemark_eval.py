#!/usr/bin/env python3
"""SAEMark evaluation on SLAM's 100-prompt benchmark.

Two modes:

  generate-detect (default)
    1. Convert SLAM prompts → JSONL
    2. Generate watermarked texts (SAEMark sentence-level FCS selection)
    3. Generate baseline texts (greedy, no watermark)
    4. Run SAEMark ROC detector on both wm and bl texts
    5. Compute cal_z = -Φ⁻¹(p_value) — same scale as SLAM's cal_z (threshold 2.0 ≈ 2.3 % FPR)
    6. Write a baselines_eval.csv row with method="saemark" (SLAM-compatible format)

  detect-attacked
    Reads SLAM's robustness_attacks.parquet (after SLAM's robustness phase 1 has run),
    runs the SAEMark detector on every saemark-method row, and writes results to
    robustness_detect.parquet so SLAM's final robustness merge picks them up.

Prerequisites:
  - conda activate slam
  - pip install -r requirements_slam_compat.txt          (bert-score, nltk extras)
  - huggingface-cli login                                 (gemma-2-2b is gated)

Usage:
  # Full generate + detect pass (resume-safe — skips stages whose output exists):
  python run_saemark_eval.py \\
      --model google/gemma-2-2b --device cuda:0 \\
      --slam-run-dir /path/to/slam/data/runs/gemma-2-2b

  # Quick smoke test (5 prompts, 5 candidates, 3 sentences):
  python run_saemark_eval.py --n-prompts 5 --candidates 5 --sentences 3 \\
      --model google/gemma-2-2b --device cuda:0

  # Detect attacked texts (after SLAM robustness phase 1):
  python run_saemark_eval.py --mode detect-attacked \\
      --model google/gemma-2-2b --device cuda:0 \\
      --slam-run-dir /path/to/slam/data/runs/gemma-2-2b
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).parent
SRC = REPO_ROOT / "src"
sys.path.insert(0, str(SRC))
SLAM_ROOT = Path(__file__).parent.parent / "slam"
sys.path.insert(0, str(SLAM_ROOT))

# Matches the actual baselines_eval.csv written by SLAM's eval_baselines stage.
BASELINES_CSV_FIELDS = [
    "prompt_id", "domain", "method",
    "cal_z", "bl_z",
    "reward_wm", "reward_bl",
    "pass", "degen",
    "prompt_text", "wm_text", "bl_text",
    "gen_time_s", "det_time_s",
]

DETECT_FIELDS = ["method", "attack", "prompt_id", "post_cal_z", "post_pass"]


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["generate-detect", "detect-attacked"],
                   default="generate-detect")
    p.add_argument("--model", default="google/gemma-2-2b",
                   help="HuggingFace ID for both the generation base and anchor model")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--n-prompts", type=int, default=100)
    p.add_argument("--candidates", type=int, default=20,
                   help="Candidate sentences per selection step (SAEMark default=50)")
    p.add_argument("--sentences", type=int, default=5,
                   help="Sentences per generated document (SAEMark default=10)")
    p.add_argument("--max-new-tokens", type=int, default=60)
    p.add_argument("--mu", type=float, default=0.13)
    p.add_argument("--sigma", type=float, default=0.02)
    p.add_argument("--target-key", default="1")
    p.add_argument("--style", choices=["completion", "instruction"], default="completion",
                   help="Prompt style: 'completion' for base models, 'instruction' for IT")
    p.add_argument("--slam-run-dir", default=None,
                   help="Path to SLAM run dir (e.g. slam/data/runs/gemma-2-2b). "
                        "If set, writes directly into that dir's results/. "
                        "If not set, writes to --output-dir.")
    p.add_argument("--output-dir", default="results/saemark_slam",
                   help="Used when --slam-run-dir is not set")
    p.add_argument("--force", action="store_true",
                   help="Overwrite existing output files")
    return p.parse_args()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def _append_jsonl(path: Path, item: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")


def _done_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {r["id"] for r in _read_jsonl(path)}


def _pvalue_to_calz(p: float, lo: float = -5.0, hi: float = 15.0) -> float:
    """Convert SAEMark p-value to SLAM-scale cal_z = -Φ⁻¹(p).

    p ≈ 0.023 → cal_z ≈ 2.0  (detection threshold, ~2.3 % FPR)
    p = 0.5   → cal_z = 0.0  (null)
    p = 1.0   → cal_z → -∞  (clipped to lo)
    """
    from scipy.stats import norm
    p_clipped = float(np.clip(p, 1e-10, 1.0 - 1e-10))
    return float(np.clip(-norm.ppf(p_clipped), lo, hi))


def _token_len(text: str, tokenizer) -> int:
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


# ── Stage: generate ───────────────────────────────────────────────────────────

def _generate_baseline_sentence(prompt: str, generate_fn, max_new_tokens: int) -> str:
    import nltk
    nltk.download("punkt_tab", quiet=True)
    completions = generate_fn(prompt, n=3, max_new_tokens=max_new_tokens)
    for comp in completions:
        sents = nltk.sent_tokenize(comp.strip())
        if sents and sents[0].endswith("."):
            return sents[0]
    # fall back: first sentence of first completion
    for comp in completions:
        sents = nltk.sent_tokenize(comp.strip())
        if sents:
            return sents[0]
    return completions[0].strip() if completions else ""


def _generate_watermarked_text(
    prompt: str, args, generate_fn, calc_ratio_fn,
    feature_mask, judge_model, judge_tokenizer, path_to_params, public_key, interval,
) -> str:
    import nltk
    from utils import generate_precise_number
    nltk.download("punkt_tab", quiet=True)

    prev_max_ind = -1
    output_text = ""
    current_prompt = prompt

    for _ in range(args.sentences):
        precise_number = generate_precise_number(
            public_key, str(prev_max_ind) if prev_max_ind != -1 else "", interval[0], interval[1]
        )
        # Collect valid sentence candidates
        candidates: list[str] = []
        for _ in range(3):
            batch = generate_fn(current_prompt, n=args.candidates, max_new_tokens=args.max_new_tokens)
            for comp in batch:
                comp = comp.strip().replace("\n", " ")
                sents = nltk.sent_tokenize(comp)
                if sents and sents[0].endswith(".") and sents[0] not in candidates:
                    candidates.append(sents[0])
            if len(candidates) >= max(5, args.candidates // 4):
                break

        if not candidates:
            break

        # Evaluate FCS for all candidates in one batch
        fcs_results = calc_ratio_fn(
            judge_model, judge_tokenizer, candidates,
            path_to_params, args.device, feature_mask,
        )
        # Select candidate closest to target pseudo-random value
        chosen_sent, (max_ind, _) = min(
            zip(candidates, fcs_results),
            key=lambda x: abs(x[1][1] - precise_number),
        )
        prev_max_ind = max_ind
        output_text = (output_text + " " + chosen_sent).strip() if output_text else chosen_sent
        current_prompt = prompt + " " + output_text

    return output_text


def _generate_baseline_text(prompt: str, args, generate_fn) -> str:
    text = ""
    current_prompt = prompt
    for _ in range(args.sentences):
        sent = _generate_baseline_sentence(current_prompt, generate_fn, args.max_new_tokens)
        if not sent:
            break
        text = (text + " " + sent).strip() if text else sent
        current_prompt = prompt + " " + text
    return text


def run_generate_detect(args: argparse.Namespace, out_dir: Path) -> None:
    import torch
    from huggingface_hub import hf_hub_download
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from fcs import calc_ratio, get_mask
    from hf_generate import init_generator, generate_batch
    from utils import get_public_key, generate_low_high_edges

    # ── Prepare SLAM prompts ───────────────────────────────────────────────
    prompts_path = out_dir / "slam_prompts.jsonl"
    if not prompts_path.exists() or args.force:
        try:
            from slam.pipeline.prompts import PROMPTS, INSTRUCTION_PROMPTS
        except ImportError:
            print("ERROR: cannot import SLAM prompts. Is /home/fabrice/slam on sys.path?")
            sys.exit(1)
        source = INSTRUCTION_PROMPTS if args.style == "instruction" else PROMPTS
        with open(prompts_path, "w", encoding="utf-8") as f:
            for i, (domain, text) in enumerate(source[: args.n_prompts]):
                # prompt_id matches SLAM's get_eval_prompts() format: "academic_000", "news_020" …
                pid = f"{domain}_{i:03d}"
                f.write(json.dumps({"id": pid, "prompt": text,
                                    "target_key": args.target_key, "domain": domain}) + "\n")
        print(f"[prepare] {len(source[:args.n_prompts])} prompts → {prompts_path}")
    else:
        print(f"[prepare] Skipping — {prompts_path} exists.")

    prompts = _read_jsonl(prompts_path)

    # ── Intermediate JSONL for generated texts ────────────────────────────
    wm_path = out_dir / "watermarked.jsonl"
    uwm_path = out_dir / "unwatermarked.jsonl"
    done_wm = _done_ids(wm_path) if not args.force else set()
    done_uwm = _done_ids(uwm_path) if not args.force else set()
    needs_generate = [p for p in prompts if p["id"] not in done_wm or p["id"] not in done_uwm]

    if needs_generate:
        print(f"\n[generate] Loading model: {args.model} …")
        judge_model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, device_map=args.device
        )
        judge_tokenizer = AutoTokenizer.from_pretrained(args.model)

        path_to_params = hf_hub_download(
            repo_id="google/gemma-scope-2b-pt-res",
            filename="layer_20/width_16k/average_l0_71/params.npz",
            force_download=False,
        )
        feature_mask = get_mask(args.device, restricted=[])
        init_generator(args.model, args.device)

        low_edge, high_edge = generate_low_high_edges(args.target_key, mu=args.mu, sigma=args.sigma)
        interval = [[low_edge, args.mu - args.sigma * 0.5], [args.mu + args.sigma * 0.5, high_edge]]
        public_key = get_public_key(args.target_key)

        for i, item in enumerate(prompts):
            pid = item["id"]
            print(f"  [{i+1}/{len(prompts)}] {pid}")

            if pid not in done_wm:
                t0 = time.perf_counter()
                wm_text = _generate_watermarked_text(
                    item["prompt"], args, generate_batch, calc_ratio,
                    feature_mask, judge_model, judge_tokenizer, path_to_params,
                    public_key, interval,
                )
                gen_time = time.perf_counter() - t0
                _append_jsonl(wm_path, {**item, "watermarked": wm_text, "gen_time_s": round(gen_time, 3)})

            if pid not in done_uwm:
                uwm_text = _generate_baseline_text(item["prompt"], args, generate_batch)
                _append_jsonl(uwm_path, {**item, "unwatermarked": uwm_text})

        # Free generation model before detection
        del judge_model, judge_tokenizer
        import gc
        gc.collect()
        torch.cuda.empty_cache()
    else:
        print(f"[generate] Skipping — all {len(prompts)} prompts already generated.")

    # ── Detection ─────────────────────────────────────────────────────────
    baselines_csv = out_dir / "baselines_eval.csv"
    if args.force and baselines_csv.exists():
        baselines_csv.unlink()

    # Load already-detected prompt IDs
    done_detect: set[str] = set()
    if baselines_csv.exists():
        import csv as _csv
        with open(baselines_csv) as f:
            done_detect = {row["prompt_id"] for row in _csv.DictReader(f) if row["method"] == "saemark"}

    wm_items = {r["id"]: r for r in _read_jsonl(wm_path)}
    uwm_items = {r["id"]: r for r in _read_jsonl(uwm_path)}
    needs_detect = [p for p in prompts if p["id"] not in done_detect and p["id"] in wm_items]

    if needs_detect:
        print(f"\n[detect] Loading judge model for {len(needs_detect)} prompts …")
        import torch
        from huggingface_hub import hf_hub_download
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from fcs import get_mask
        from utils import get_public_key, generate_low_high_edges

        judge_model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, device_map=args.device
        )
        judge_tokenizer = AutoTokenizer.from_pretrained(args.model)
        path_to_params = hf_hub_download(
            repo_id="google/gemma-scope-2b-pt-res",
            filename="layer_20/width_16k/average_l0_71/params.npz",
            force_download=False,
        )
        feature_mask = get_mask(args.device, restricted=[])

        from validator_operations.roc_score import validate_roc_en

        low_edge, high_edge = generate_low_high_edges(args.target_key, mu=args.mu, sigma=args.sigma)
        intervals = [[[low_edge, args.mu - args.sigma * 0.5], [args.mu + args.sigma * 0.5, high_edge]]]
        public_keys = [get_public_key(args.target_key)]

        write_header = not baselines_csv.exists()
        with open(baselines_csv, "a", newline="") as cf:
            writer = csv.DictWriter(cf, fieldnames=BASELINES_CSV_FIELDS)
            if write_header:
                writer.writeheader()

            for i, item in enumerate(needs_detect):
                pid = item["id"]
                wm_rec = wm_items[pid]
                uwm_rec = uwm_items.get(pid, {})
                wm_text = wm_rec.get("watermarked", "")
                bl_text = uwm_rec.get("unwatermarked", "")

                # Detect watermarked text
                t0 = time.perf_counter()
                p_wm = validate_roc_en(
                    wm_text, public_keys, args.device, intervals,
                    judge_model, judge_tokenizer, path_to_params, feature_mask,
                )
                # Detect baseline text (to get bl_z)
                p_bl = validate_roc_en(
                    bl_text, public_keys, args.device, intervals,
                    judge_model, judge_tokenizer, path_to_params, feature_mask,
                ) if bl_text else 1.0
                det_time = time.perf_counter() - t0

                cal_z = _pvalue_to_calz(p_wm)
                bl_z = _pvalue_to_calz(p_bl)
                gen_time = wm_rec.get("gen_time_s", float("nan"))

                row = {
                    "prompt_id": pid,
                    "domain": item.get("domain", ""),
                    "method": "saemark",
                    "cal_z": round(cal_z, 4),
                    "bl_z": round(bl_z, 4),
                    "reward_wm": "",   # filled by SLAM's score_quality stage
                    "reward_bl": "",
                    "pass": int(cal_z > 2.0),
                    "degen": 0,
                    "prompt_text": item["prompt"],
                    "wm_text": wm_text,
                    "bl_text": bl_text,
                    "gen_time_s": round(gen_time, 3) if not np.isnan(gen_time) else "",
                    "det_time_s": round(det_time, 3),
                }
                writer.writerow(row)
                cf.flush()
                print(f"  [{i+1}/{len(needs_detect)}] {pid}  cal_z={cal_z:.3f}  pass={row['pass']}")
    else:
        print(f"[detect] Skipping — {len(done_detect)} prompts already in {baselines_csv}")

    # ── Summary ────────────────────────────────────────────────────────────
    import pandas as pd
    df = pd.read_csv(baselines_csv)
    saemark = df[df["method"] == "saemark"]
    tpr = float((saemark["cal_z"] > 2.0).mean())
    print(f"\n[result] saemark  n={len(saemark)}  TPR@2.3%FPR={tpr:.3f}  "
          f"mean_cal_z={saemark['cal_z'].mean():.3f}")
    print(f"[result] CSV → {baselines_csv}")


# ── Stage: detect-attacked ────────────────────────────────────────────────────

def run_detect_attacked(args: argparse.Namespace, results_dir: Path) -> None:
    """Read SLAM's robustness_attacks.parquet, detect saemark rows, write to robustness_detect.parquet."""
    import pandas as pd
    import torch
    from huggingface_hub import hf_hub_download
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from fcs import get_mask
    from utils import get_public_key, generate_low_high_edges
    from validator_operations.roc_score import validate_roc_en

    attacks_path = results_dir / "robustness_attacks.parquet"
    detect_path = results_dir / "robustness_detect.parquet"

    if not attacks_path.exists():
        print(f"ERROR: {attacks_path} not found. Run SLAM's robustness phase 1 first.")
        sys.exit(1)

    attacks_df = pd.read_parquet(attacks_path)
    saemark_attacks = attacks_df[attacks_df["method"] == "saemark"]
    if saemark_attacks.empty:
        print("No saemark rows in robustness_attacks.parquet — nothing to do.")
        return

    existing = pd.read_parquet(detect_path) if detect_path.exists() else pd.DataFrame()
    done = set(zip(existing.get("method", []), existing.get("attack", []), existing.get("prompt_id", [])))

    pending = saemark_attacks[
        ~saemark_attacks.apply(lambda r: (r["method"], r["attack"], r["prompt_id"]) in done, axis=1)
    ]
    if pending.empty:
        print(f"[detect-attacked] All {len(saemark_attacks)} saemark rows already scored.")
        return

    print(f"[detect-attacked] Loading judge model for {len(pending)} pending texts …")
    judge_model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map=args.device
    )
    judge_tokenizer = AutoTokenizer.from_pretrained(args.model)
    path_to_params = hf_hub_download(
        repo_id="google/gemma-scope-2b-pt-res",
        filename="layer_20/width_16k/average_l0_71/params.npz",
        force_download=False,
    )
    feature_mask = get_mask(args.device, restricted=[])
    low_edge, high_edge = generate_low_high_edges(args.target_key, mu=args.mu, sigma=args.sigma)
    intervals = [[[low_edge, args.mu - args.sigma * 0.5], [args.mu + args.sigma * 0.5, high_edge]]]
    public_keys = [get_public_key(args.target_key)]

    new_rows: list[dict] = []
    for i, (_, row) in enumerate(pending.iterrows()):
        text = row["attacked_text"]
        if not text or (isinstance(text, float) and np.isnan(text)):
            p = 1.0
        else:
            p = validate_roc_en(
                str(text), public_keys, args.device, intervals,
                judge_model, judge_tokenizer, path_to_params, feature_mask,
            )
        cal_z = _pvalue_to_calz(p)
        new_rows.append({
            "method": "saemark",
            "attack": row["attack"],
            "prompt_id": row["prompt_id"],
            "post_cal_z": round(cal_z, 4),
            "post_pass": int(cal_z > 2.0),
        })
        print(f"  [{i+1}/{len(pending)}] {row['attack']}:{row['prompt_id']}  cal_z={cal_z:.3f}")

    merged = pd.concat(
        [existing, pd.DataFrame(new_rows)], ignore_index=True
    ).drop_duplicates(subset=["method", "attack", "prompt_id"])
    merged.to_parquet(detect_path, index=False)
    print(f"\n[detect-attacked] {len(new_rows)} rows → {detect_path}")

    # Summary by attack type
    sm = merged[merged["method"] == "saemark"]
    for atk, grp in sm.groupby("attack"):
        tpr = float(grp["post_pass"].mean())
        print(f"  {atk:25s}  TPR={tpr:.3f}  (n={len(grp)})")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    if args.slam_run_dir:
        results_dir = Path(args.slam_run_dir) / "results"
    else:
        results_dir = Path(args.output_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"SAEMark eval — mode={args.mode}  model={args.model}  device={args.device}")
    print(f"  results dir: {results_dir}\n")

    if args.mode == "generate-detect":
        run_generate_detect(args, results_dir)
    else:
        run_detect_attacked(args, results_dir)


if __name__ == "__main__":
    main()
