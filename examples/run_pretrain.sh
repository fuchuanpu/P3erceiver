#!/usr/bin/env bash
# Pre-train the encoder from a raw backbone trace (sampling-invariant
# objectives; no labels). Reproduces ckpt/pretrained.pt. ~30 min on one GPU.
#
# The trace is NOT shipped -- see data/README.md for the download URL.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export P3_EST_IAT=1
PY_BIN="${PYTHON:-python}"
command -v "$PY_BIN" >/dev/null || PY_BIN=python3
command -v "$PY_BIN" >/dev/null || { echo "no python on PATH -- run: conda activate p3" >&2; exit 1; }

TRACE="${1:-}"
if [[ -z "$TRACE" ]]; then
  echo "usage: bash examples/run_pretrain.sh /path/to/backbone.pcap" >&2
  exit 1
fi

"$PY_BIN" p3/pretrain.py --pcap "$TRACE" --tag backbone_repro
