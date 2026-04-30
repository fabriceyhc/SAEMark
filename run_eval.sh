#!/usr/bin/env bash
# SAEMark evaluation on SLAM's 100-prompt benchmark.
#
# Prerequisites:
#   1. huggingface-cli login  (google/gemma-2-2b is gated)
#   2. conda activate slam
#   3. pip install -r requirements_slam_compat.txt
#
# Quick smoke test (5 prompts):
#   N_PROMPTS=5 CANDIDATES=5 SENTENCES=3 bash run_eval.sh
#
# Full run (writes directly into a SLAM run dir):
#   SLAM_RUN_DIR=/home/fabrice/slam/data/runs/gemma-2-2b bash run_eval.sh
#
# After SLAM's robustness phase 1 has generated attacked texts:
#   bash run_eval.sh --mode detect-attacked

set -euo pipefail
REPO="$(cd "$(dirname "$0")" && pwd)"

DEVICE="${DEVICE:-cuda:0}"
MODEL="${MODEL:-google/gemma-2-2b}"
N_PROMPTS="${N_PROMPTS:-100}"
CANDIDATES="${CANDIDATES:-20}"
SENTENCES="${SENTENCES:-5}"
SLAM_RUN_DIR="${SLAM_RUN_DIR:-}"

if [[ "${CONDA_DEFAULT_ENV:-}" != "slam" ]]; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate slam
fi
pip install -q -r "${REPO}/requirements_slam_compat.txt"

EXTRA_ARGS=()
if [[ -n "${SLAM_RUN_DIR}" ]]; then
    EXTRA_ARGS+=(--slam-run-dir "${SLAM_RUN_DIR}")
fi

python "${REPO}/run_saemark_eval.py" \
    --model "${MODEL}" \
    --device "${DEVICE}" \
    --n-prompts "${N_PROMPTS}" \
    --candidates "${CANDIDATES}" \
    --sentences "${SENTENCES}" \
    "${EXTRA_ARGS[@]}" \
    "$@"
