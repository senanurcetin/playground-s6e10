#!/usr/bin/env bash
# Launch day: download -> check -> fast LightGBM baseline -> submission (after you joined the competition).
#   KAGGLE_API_TOKEN=... bash launch.sh [competition-slug]
set -euo pipefail
cd "$(dirname "$0")"
COMP=${1:-playground-series-s6e10}
export COMP DATA="C:/Users/MONSTER/kaggle-data/$COMP"
PY=../enveda-casmi26/.venv/Scripts/python
mkdir -p "$DATA"
timeout 600 kaggle competitions download -c "$COMP" -p "$DATA"
(cd "$DATA" && unzip -o -q "$COMP.zip")
ls "$DATA"
$PY pipeline.py check
$PY pipeline.py train --models lgb
$PY pipeline.py blend
echo "baseline ready: work/submission.csv — submit with:"
echo "  kaggle competitions submit -c $COMP -f work/submission.csv -m 'LGB baseline, 5-fold CV'"
