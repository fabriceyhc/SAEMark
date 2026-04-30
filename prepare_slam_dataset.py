#!/usr/bin/env python3
"""Convert SLAM's canonical 100-prompt eval set to SAEMark's JSONL format.

Output format (one JSON object per line):
    {"id": "slam/1", "prompt": "<text>", "target_key": "1", "domain": "<domain>"}

Usage:
    python prepare_slam_dataset.py --output data/slam_prompts.jsonl
    python prepare_slam_dataset.py --style instruction --output data/slam_prompts_it.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "slam"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/slam_prompts.jsonl")
    parser.add_argument(
        "--style",
        choices=["completion", "instruction"],
        default="completion",
        help="'completion' for base models, 'instruction' for IT models",
    )
    parser.add_argument("--target-key", default="1", help="Watermark key for all prompts")
    parser.add_argument("--n", type=int, default=100, help="Number of prompts to include")
    args = parser.parse_args()

    try:
        from slam.pipeline.prompts import PROMPTS, INSTRUCTION_PROMPTS
    except ImportError:
        print("ERROR: Could not import SLAM prompts. Ensure /home/fabrice/slam is accessible.")
        sys.exit(1)

    source = INSTRUCTION_PROMPTS if args.style == "instruction" else PROMPTS
    prompts = source[: args.n]

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        for i, (domain, text) in enumerate(prompts):
            item = {
                "id": f"slam/{i + 1}",
                "prompt": text,
                "target_key": args.target_key,
                "domain": domain,
            }
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Wrote {len(prompts)} prompts ({args.style}) → {out_path}")


if __name__ == "__main__":
    main()
