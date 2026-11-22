#!/usr/bin/env bash
# Run one experiment end to end: dataset, CLIP features, training, evaluation.
#
#   scripts/run_experiment.sh configs/mlp_frozen.yaml [--resume]
#
# Every stage is idempotent: the dataset and the features are reused when
# present, and --resume continues an interrupted training run.
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 || ( $# -eq 2 && "$2" != "--resume" ) ]]; then
    echo "usage: $0 <config.yaml> [--resume]" >&2
    exit 64
fi
config=$1
resume=${2:-}

if [[ ! -f "$config" ]]; then
    echo "error: configuration file not found: $config" >&2
    exit 66
fi

# Paths inside the configuration are relative to the repository root, so run
# from there; the configuration itself may be given relative to the caller.
config=$(realpath "$config")
cd "$(dirname "$0")/.."

if [[ ! -x .venv/bin/clipcap ]]; then
    python3 -m venv .venv
    .venv/bin/pip install --quiet --upgrade pip
    .venv/bin/pip install --quiet -e .
fi
clipcap=.venv/bin/clipcap

# A stale Hugging Face token makes public downloads fail with 401.
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1

meteor=()
if command -v java >/dev/null 2>&1; then
    meteor=(--meteor)
fi

"$clipcap" download --config "$config"
"$clipcap" extract --config "$config"
if [[ -n "$resume" ]]; then
    "$clipcap" train --config "$config" --resume
else
    "$clipcap" train --config "$config"
fi
"$clipcap" evaluate --config "$config" --split val "${meteor[@]}"
"$clipcap" evaluate --config "$config" --split test "${meteor[@]}"
