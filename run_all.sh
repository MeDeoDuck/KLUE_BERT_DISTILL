#!/usr/bin/env bash
# One-shot: dataset prep → training → evaluation.
# Run from repo root after `pip install -r requirements.txt`.

set -euo pipefail

cd "$(dirname "$0")"

echo "============================================================"
echo "[1/3] prepare_dataset"
echo "============================================================"
python -m local_classifier.prepare_dataset

echo
echo "============================================================"
echo "[2/3] train"
echo "============================================================"
python -m local_classifier.train

echo
echo "============================================================"
echo "[3/3] evaluate"
echo "============================================================"
python -m local_classifier.evaluate

echo
echo "done. artifacts → local_classifier/artifacts/"
