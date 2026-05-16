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

# Minimal fallback column set — used only when baselines_eval.csv doesn't exist yet.
# If the file already exists, _get_csv_fields() reads its header instead so
# appended rows stay aligned regardless of which SLAM pipeline version wrote it.
BASELINES_CSV_FIELDS_DEFAULT = [
    "prompt_id", "domain", "method",
    "cal_z", "bl_z",
    "reward_wm", "reward_bl",
    "pass", "degen",
    "prompt_text", "wm_text", "bl_text",
    "ppl_wm", "ppl_bl", "ppl_ratio",
]


def _get_csv_fields(path: Path) -> list[str]:
    """Return column names from an existing CSV header, else the default fallback."""
    if path.exists():
        import csv as _csv
        with open(path, newline="") as f:
            reader = _csv.reader(f)
            header = next(reader, None)
        if header:
            return header
    return BASELINES_CSV_FIELDS_DEFAULT

DETECT_FIELDS = ["method", "attack", "prompt_id", "post_cal_z", "post_pass"]


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["generate-detect", "detect-attacked"],
                   default="generate-detect")
    # Base (generation) model — can be any supported model.
    p.add_argument("--model", default="google/gemma-2-2b",
                   help="HuggingFace ID of the base generation model")
    # Anchor model — defaults to same as base model; SAE is selected to match.
    p.add_argument("--anchor", default=None,
                   help="HuggingFace ID of the anchor model for FCS scoring. "
                        "Defaults to --model. Must be a GemmaScope-supported model "
                        "(gemma-2-2b or gemma-2-9b) so the SAE dimensions match.")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--n-prompts", type=int, default=100)
    p.add_argument("--candidates", type=int, default=50,
                   help="Candidate sentences per selection step (paper default=50)")
    p.add_argument("--sentences", type=int, default=10,
                   help="Sentences per generated document (paper default=10)")
    p.add_argument("--max-new-tokens", type=int, default=60)
    p.add_argument("--mu", type=float, default=0.13)
    p.add_argument("--sigma", type=float, default=0.02)
    p.add_argument("--rmin", type=float, default=0.95,
                   help="Min rank1 for document quality acceptance (paper default=0.95)")
    p.add_argument("--rmax", type=float, default=1.05,
                   help="Max rank1 for document quality acceptance (paper default=1.05)")
    p.add_argument("--omin", type=float, default=0.95,
                   help="Min rank2 (overlap) for acceptance (paper default=0.95)")
    p.add_argument("--attempts", type=int, default=5,
                   help="Max retries per document if quality thresholds not met")
    p.add_argument("--mask", action="store_true", default=True,
                   help="Apply background frequent-feature mask (paper default=True)")
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


# SAE configs keyed by anchor model short name.
# SAE dimensions must match the anchor model's hidden size.
_SAE_CONFIGS = {
    "gemma-2-2b":    ("google/gemma-scope-2b-pt-res", "layer_20/width_16k/average_l0_71/params.npz", 20),
    "gemma-2-2b-it": ("google/gemma-scope-2b-pt-res", "layer_20/width_16k/average_l0_71/params.npz", 20),
    "gemma-2-9b":    ("google/gemma-scope-9b-pt-res", "layer_20/width_16k/average_l0_68/params.npz", 20),
    "gemma-2-9b-it": ("google/gemma-scope-9b-pt-res", "layer_20/width_16k/average_l0_68/params.npz", 20),
}


def _resolve_sae(anchor_id: str) -> tuple[str, str, int]:
    short = anchor_id.split("/")[-1]
    if short not in _SAE_CONFIGS:
        raise ValueError(f"No SAE config for anchor '{anchor_id}'. "
                         f"Supported: {list(_SAE_CONFIGS)}")
    return _SAE_CONFIGS[short]


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


def _write_saemark_timing(
    out_dir: Path,
    model_id: str,
    gen_times: list[float],
    det_times: list[float],
    wm_tok_lens: list[int],
) -> None:
    """Aggregate per-prompt timing and write/merge into bench_timing JSON.

    Merges saemark results into an existing bench_timing_{model}.json produced
    by SLAM's bench_timing.py so that table_timing.py can include SAEMark in
    the same LaTeX table row.  If no bench_timing JSON exists, writes a
    standalone saemark_timing.json in the results dir instead.

    Fields written match bench_timing.py's output schema:
      gen_mean, gen_std  — seconds per document (generation only)
      det_mean, det_std  — seconds per document (detection only)
      tok_per_s_gen      — generated tokens / second (length-adjusted gen throughput)
      n                  — number of samples
    """
    gen_arr = np.array(gen_times, dtype=np.float64)
    det_arr = np.array(det_times, dtype=np.float64)
    tok_arr = np.array(wm_tok_lens, dtype=np.float64)

    entry = {
        "gen_mean":     float(gen_arr.mean()),
        "gen_std":      float(gen_arr.std()),
        "det_mean":     float(det_arr.mean()),
        "det_std":      float(det_arr.std()),
        "tok_per_s_gen": float(tok_arr.mean() / gen_arr.mean()) if gen_arr.mean() > 0 else 0.0,
        "n":            len(gen_times),
    }

    # Try to merge into an existing bench_timing JSON in the SLAM artifacts dir.
    # File names use the short model name (e.g. "gemma-2-2b"), not the full HF ID.
    slam_artifacts = SLAM_ROOT / "artifacts" / "tables"
    model_tag = model_id.split("/")[-1]
    bench_json = slam_artifacts / f"bench_timing_{model_tag}.json"

    if bench_json.exists():
        import json as _json
        with open(bench_json) as f:
            data = _json.load(f)
        data.setdefault("results", {})["saemark"] = entry
        with open(bench_json, "w") as f:
            _json.dump(data, f, indent=2)
        print(f"[timing] Merged saemark timing into {bench_json}")
    else:
        # Write standalone file in results dir
        import json as _json
        standalone = out_dir / "saemark_timing.json"
        payload = {"model": model_id, "results": {"saemark": entry}}
        with open(standalone, "w") as f:
            _json.dump(payload, f, indent=2)
        print(f"[timing] No bench_timing JSON found at {bench_json}; "
              f"wrote standalone timing to {standalone}")

    print(f"[timing] saemark  n={entry['n']}  "
          f"gen={entry['gen_mean']:.2f}±{entry['gen_std']:.2f}s  "
          f"det={entry['det_mean']*1000:.0f}±{entry['det_std']*1000:.0f}ms  "
          f"tok/s={entry['tok_per_s_gen']:.1f}")


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
    prompt: str,
    args,
    generate_fn,
    calc_ratio_fn,
    feature_mask,
    judge_model,
    judge_tokenizer,
    path_to_params,
    public_key,
    interval,
    sae_layer: int = 20,
) -> tuple[str, float, float]:
    """Generate one watermarked document, returning (text, rank1, rank2).

    Faithfully ports generate_paragraph_en() + the quality checks from
    watermarked_generator.py:
      rank1 = spread(actual FCS) / spread(target pseudo-random)  ≈ 1.0 = good
      rank2 = overlap(actual, target) / spread(target)           ≥ 0.95 = good
    Callers should retry up to args.attempts times if thresholds not met.
    """
    import nltk
    from utils import generate_precise_number
    nltk.download("punkt_tab", quiet=True)

    prev_max_ind = -1
    output_text = ""
    current_prompt = prompt
    precise_numbers: list[float] = []
    actual_numbers: list[float] = []

    for _ in range(args.sentences):
        seed = str(prev_max_ind) if prev_max_ind != -1 else ""
        precise_number = generate_precise_number(public_key, seed, interval[0], interval[1])

        # Collect valid sentence candidates (up to 3 batches to reach target count)
        candidates: list[str] = []
        for _ in range(3):
            batch = generate_fn(current_prompt, n=args.candidates, max_new_tokens=args.max_new_tokens)
            for comp in batch:
                comp = comp.strip().replace("\n", " ")
                sents = nltk.sent_tokenize(comp)
                if sents and sents[0].endswith(".") and sents[0] not in candidates:
                    candidates.append(sents[0])
            if len(candidates) >= args.candidates:
                break
        if len(candidates) > args.candidates:
            candidates = candidates[:args.candidates]

        if not candidates:
            break

        # Score all candidates with the anchor model FCS
        fcs_results = calc_ratio_fn(
            judge_model, judge_tokenizer, candidates,
            path_to_params, args.device, feature_mask,
            target_layer=sae_layer,
        )
        # Select candidate whose FCS ratio is closest to the target pseudo-random value
        ranked = sorted(
            zip(candidates, fcs_results),
            key=lambda x: abs(x[1][1] - precise_number),
        )
        chosen_sent, (max_ind, chosen_ratio) = ranked[0]

        prev_max_ind = max_ind
        precise_numbers.append(precise_number)
        actual_numbers.append(chosen_ratio)
        output_text = (output_text + " " + chosen_sent).strip() if output_text else chosen_sent
        current_prompt = prompt + " " + output_text

    if len(actual_numbers) < 2:
        return output_text, 0.0, 0.0

    actual_min, actual_max = min(actual_numbers), max(actual_numbers)
    precise_min, precise_max = min(precise_numbers), max(precise_numbers)
    spread = precise_max - precise_min
    if spread == 0:
        return output_text, 0.0, 0.0

    rank1 = (actual_max - actual_min) / spread
    cross_min = max(actual_min, precise_min)
    cross_max = min(actual_max, precise_max)
    rank2 = (cross_max - cross_min) / spread
    return output_text, rank1, rank2


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

    # Anchor defaults to base model; SAE selected to match anchor's hidden size.
    if args.anchor is None:
        args.anchor = args.model
    SAE_REPO, SAE_FILE, SAE_LAYER = _resolve_sae(args.anchor)
    print(f"[config] base={args.model}  anchor={args.anchor}  "
          f"sae={SAE_REPO}/{SAE_FILE}  layer={SAE_LAYER}")
    print(f"[config] candidates={args.candidates}  sentences={args.sentences}  "
          f"attempts={args.attempts}  mask={args.mask}")

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
        anchor_differs = args.anchor != args.model
        if anchor_differs:
            print(f"\n[generate] Loading anchor model: {args.anchor} …")
        else:
            print(f"\n[generate] Loading model (shared for generation + FCS): {args.model} …")

        judge_model = AutoModelForCausalLM.from_pretrained(
            args.anchor, torch_dtype=torch.bfloat16, device_map=args.device
        )
        judge_tokenizer = AutoTokenizer.from_pretrained(args.anchor)

        path_to_params = hf_hub_download(
            repo_id=SAE_REPO, filename=SAE_FILE, force_download=False,
        )
        # Background frequent-feature mask (paper default: mask=True, lang=en)
        if args.mask:
            import json as _json
            bg_path = REPO_ROOT / "src" / "dataset" / "bg_freq_feat_mask" / "original.json"
            with open(bg_path) as f:
                bg_data = [_json.loads(l) for l in f if l.strip()]
            restricted = next((d["list"] for d in bg_data if d["lang"] == "en"), [])
        else:
            restricted = []
        feature_mask = get_mask(args.device, restricted=restricted)
        print(f"[generate] Feature mask: {len(restricted)} restricted features")

        if anchor_differs:
            # anchor != base: load base separately and use hf_generate singleton
            print(f"[generate] Loading base model: {args.model} …")
            init_generator(args.model, args.device)
            gen_fn = generate_batch
        else:
            # anchor == base: reuse judge_model for generation to avoid double-loading
            # (critical for 9b: two 18GB models on one 40GB GPU = OOM)
            import hf_generate as _hfg
            _hfg._model = judge_model
            _hfg._tokenizer = AutoTokenizer.from_pretrained(args.model)
            _hfg._tokenizer.padding_side = "left"
            if _hfg._tokenizer.pad_token is None:
                _hfg._tokenizer.pad_token = _hfg._tokenizer.eos_token
            _hfg._model_path = args.model
            judge_model.eval()
            gen_fn = generate_batch
            print(f"[generate] Shared model for generation + FCS (anchor == base)")

        low_edge, high_edge = generate_low_high_edges(args.target_key, mu=args.mu, sigma=args.sigma)
        interval = [[low_edge, args.mu - args.sigma * 0.5], [args.mu + args.sigma * 0.5, high_edge]]
        public_key = get_public_key(args.target_key)

        for i, item in enumerate(prompts):
            pid = item["id"]
            print(f"  [{i+1}/{len(prompts)}] {pid}", end="", flush=True)

            if pid not in done_wm:
                t0 = time.perf_counter()
                wm_text, rank1, rank2 = "", 0.0, 0.0
                for attempt in range(args.attempts):
                    wm_text, rank1, rank2 = _generate_watermarked_text(
                        item["prompt"], args, gen_fn, calc_ratio,
                        feature_mask, judge_model, judge_tokenizer, path_to_params,
                        public_key, interval, sae_layer=SAE_LAYER,
                    )
                    if args.rmin < rank1 < args.rmax and rank2 >= args.omin:
                        break
                gen_time = time.perf_counter() - t0
                print(f"  rank1={rank1:.3f} rank2={rank2:.3f} gen={gen_time:.1f}s")
                _append_jsonl(wm_path, {**item, "watermarked": wm_text,
                                        "gen_time_s": round(gen_time, 3),
                                        "rank1": round(rank1, 4),
                                        "rank2": round(rank2, 4)})
            else:
                print()

            if pid not in done_uwm:
                uwm_text = _generate_baseline_text(item["prompt"], args, gen_fn)
                _append_jsonl(uwm_path, {**item, "unwatermarked": uwm_text})

        # Free models before detection stage
        import hf_generate as _hfg
        _hfg._model = None
        _hfg._tokenizer = None
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
        print(f"\n[detect] Loading anchor model ({args.anchor}) for {len(needs_detect)} prompts …")
        import torch
        from huggingface_hub import hf_hub_download
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from fcs import get_mask
        from utils import get_public_key, generate_low_high_edges

        judge_model = AutoModelForCausalLM.from_pretrained(
            args.anchor, torch_dtype=torch.bfloat16, device_map=args.device
        )
        judge_tokenizer = AutoTokenizer.from_pretrained(args.anchor)
        path_to_params = hf_hub_download(
            repo_id=SAE_REPO, filename=SAE_FILE, force_download=False,
        )
        if args.mask:
            import json as _json
            bg_path = REPO_ROOT / "src" / "dataset" / "bg_freq_feat_mask" / "original.json"
            with open(bg_path) as f:
                bg_data = [_json.loads(l) for l in f if l.strip()]
            restricted = next((d["list"] for d in bg_data if d["lang"] == "en"), [])
        else:
            restricted = []
        feature_mask = get_mask(args.device, restricted=restricted)

        from validator_operations.roc_score import validate_roc_en

        low_edge, high_edge = generate_low_high_edges(args.target_key, mu=args.mu, sigma=args.sigma)
        intervals = [[[low_edge, args.mu - args.sigma * 0.5], [args.mu + args.sigma * 0.5, high_edge]]]
        public_keys = [get_public_key(args.target_key)]

        # Per-prompt timing accumulators for bench_timing JSON
        _gen_times: list[float] = []
        _det_times: list[float] = []
        _wm_tok_lens: list[int] = []

        # Read existing header so appended rows are column-aligned regardless of
        # which SLAM pipeline version wrote the file (column sets differ across models).
        csv_fields = _get_csv_fields(baselines_csv)
        write_header = not baselines_csv.exists()
        with open(baselines_csv, "a", newline="") as cf:
            writer = csv.DictWriter(cf, fieldnames=csv_fields, extrasaction="ignore")
            if write_header:
                writer.writeheader()

            for i, item in enumerate(needs_detect):
                pid = item["id"]
                wm_rec = wm_items[pid]
                uwm_rec = uwm_items.get(pid, {})
                wm_text = wm_rec.get("watermarked", "")
                bl_text = uwm_rec.get("unwatermarked", "")
                gen_time_s = float(wm_rec.get("gen_time_s", float("nan")))
                wm_tok_len = _token_len(wm_text, judge_tokenizer) if wm_text else 0

                # Detect watermarked text, timed for timing-table export
                t_det0 = time.perf_counter()
                p_wm = validate_roc_en(
                    wm_text, public_keys, args.device, intervals,
                    judge_model, judge_tokenizer, path_to_params, feature_mask,
                    target_layer=SAE_LAYER,
                )
                det_time_s = time.perf_counter() - t_det0
                # Detect baseline text (to get bl_z); not included in det timing
                p_bl = validate_roc_en(
                    bl_text, public_keys, args.device, intervals,
                    judge_model, judge_tokenizer, path_to_params, feature_mask,
                    target_layer=SAE_LAYER,
                ) if bl_text else 1.0

                cal_z = _pvalue_to_calz(p_wm)
                bl_z = _pvalue_to_calz(p_bl)

                # Accumulate timing samples (skip NaN gen times from resumed runs)
                if not np.isnan(gen_time_s):
                    _gen_times.append(gen_time_s)
                    _det_times.append(det_time_s)
                    _wm_tok_lens.append(wm_tok_len)

                # Superset of all known column variants; DictWriter ignores extras.
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
                    "gen_time_s": round(gen_time_s, 3) if not np.isnan(gen_time_s) else "",
                    "wm_cont_len": wm_tok_len,
                    "bl_cont_len": _token_len(bl_text, judge_tokenizer) if bl_text else "",
                    "prompt_text": item["prompt"],
                    "wm_text": wm_text,
                    "bl_text": bl_text,
                    "ppl_wm": "",      # filled by SLAM's eval_quality stage
                    "ppl_bl": "",
                    "ppl_ratio": "",
                }
                writer.writerow(row)
                cf.flush()
                print(f"  [{i+1}/{len(needs_detect)}] {pid}  cal_z={cal_z:.3f}  pass={row['pass']}"
                      f"  gen={gen_time_s:.1f}s  det={det_time_s*1000:.0f}ms")

        # Write aggregated timing data compatible with bench_timing.py JSON format
        if _gen_times:
            _write_saemark_timing(out_dir, args.model, _gen_times, _det_times, _wm_tok_lens)
    else:
        print(f"[detect] Skipping — {len(done_detect)} prompts already in {baselines_csv}")

    # ── Write run config so detect-attacked knows which anchor was used ────
    import json as _json
    run_cfg_path = out_dir / "saemark_run_config.json"
    with open(run_cfg_path, "w") as _f:
        _json.dump({"model": args.model, "anchor": args.anchor}, _f, indent=2)
    print(f"[config] Wrote run config → {run_cfg_path}")

    # ── Summary ────────────────────────────────────────────────────────────
    import pandas as pd
    df = pd.read_csv(baselines_csv)
    saemark = df[df["method"] == "saemark"]
    tpr = float((saemark["cal_z"] > 2.0).mean())
    print(f"\n[result] saemark  n={len(saemark)}  TPR@2.3%FPR={tpr:.3f}  "
          f"mean_cal_z={saemark['cal_z'].mean():.3f}")
    print(f"[result] CSV → {baselines_csv}")


# ── Stage: detect-attacked ────────────────────────────────────────────────────

def _validate_with_timeout(text, public_keys, device, intervals,
                           judge_model, judge_tokenizer, path_to_params,
                           feature_mask, target_layer, timeout_sec=120):
    """Run validate_roc_en with a SIGALRM-based timeout. Returns p=1.0 on timeout."""
    import signal
    from validator_operations.roc_score import validate_roc_en

    def _handler(signum, frame):
        raise TimeoutError("validate_roc_en timed out")

    old = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(timeout_sec)
    try:
        return validate_roc_en(
            text, public_keys, device, intervals,
            judge_model, judge_tokenizer, path_to_params, feature_mask,
            target_layer=target_layer,
        )
    except TimeoutError:
        return 1.0
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


def run_detect_attacked(args: argparse.Namespace, results_dir: Path) -> None:
    """Read SLAM's robustness_attacks.parquet, detect saemark rows, write to robustness_detect.parquet."""
    import pandas as pd
    import torch
    from huggingface_hub import hf_hub_download
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from fcs import get_mask
    from utils import get_public_key, generate_low_high_edges

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

    if args.anchor is None:
        args.anchor = args.model
    SAE_REPO, SAE_FILE, SAE_LAYER = _resolve_sae(args.anchor)
    print(f"[detect-attacked] anchor={args.anchor}  sae={SAE_REPO}/{SAE_FILE}")
    print(f"[detect-attacked] Loading anchor model for {len(pending)} pending texts …")
    judge_model = AutoModelForCausalLM.from_pretrained(
        args.anchor, torch_dtype=torch.bfloat16, device_map=args.device
    )
    judge_tokenizer = AutoTokenizer.from_pretrained(args.anchor)
    path_to_params = hf_hub_download(
        repo_id=SAE_REPO, filename=SAE_FILE, force_download=False,
    )
    if args.mask:
        import json as _json
        bg_path = REPO_ROOT / "src" / "dataset" / "bg_freq_feat_mask" / "original.json"
        with open(bg_path) as f:
            bg_data = [_json.loads(l) for l in f if l.strip()]
        restricted = next((d["list"] for d in bg_data if d["lang"] == "en"), [])
    else:
        restricted = []
    feature_mask = get_mask(args.device, restricted=restricted)
    low_edge, high_edge = generate_low_high_edges(args.target_key, mu=args.mu, sigma=args.sigma)
    intervals = [[[low_edge, args.mu - args.sigma * 0.5], [args.mu + args.sigma * 0.5, high_edge]]]
    public_keys = [get_public_key(args.target_key)]

    n_total = len(saemark_attacks)
    n_done_start = len(existing)
    for i, (_, row) in enumerate(pending.iterrows()):
        text = row["attacked_text"]
        timed_out = False
        if not text or (isinstance(text, float) and np.isnan(text)):
            p = 1.0
        else:
            p = _validate_with_timeout(
                str(text), public_keys, args.device, intervals,
                judge_model, judge_tokenizer, path_to_params, feature_mask,
                target_layer=SAE_LAYER, timeout_sec=120,
            )
            timed_out = (p == 1.0 and len(str(text)) > 10)
        cal_z = _pvalue_to_calz(p)
        new_row = {
            "method": "saemark",
            "attack": row["attack"],
            "prompt_id": row["prompt_id"],
            "post_cal_z": round(cal_z, 4),
            "post_pass": int(cal_z > 2.0),
        }
        suffix = " [TIMEOUT→p=1]" if timed_out else ""
        print(f"  [{n_done_start+i+1}/{n_total}] {row['attack']}:{row['prompt_id']}  cal_z={cal_z:.3f}{suffix}", flush=True)

        # Incremental save: merge new row into parquet after each text
        existing = pd.concat(
            [existing, pd.DataFrame([new_row])], ignore_index=True
        ).drop_duplicates(subset=["method", "attack", "prompt_id"])
        existing.to_parquet(detect_path, index=False)

    print(f"\n[detect-attacked] {len(existing)} rows → {detect_path}")

    # Summary by attack type
    sm = existing[existing["method"] == "saemark"]
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
