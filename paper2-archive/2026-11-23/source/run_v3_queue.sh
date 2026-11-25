#!/usr/bin/env bash
set -euo pipefail

WORKDIR="/root/YOLO26N-FCHL/reports/paper2_cls_candidate_diagnosis_640_20261120_42106"
PY="/opt/conda/bin/python"
cd "$WORKDIR"
mkdir -p logs

exec 9>/tmp/paper2_v3_queue.lock
flock -n 9 || { echo "Another paper2_v3_queue is already running!"; exit 1; }

run_arm() {
  local arm="$1"
  echo "[$(date -Iseconds)] === START $arm ==="
  "$PY" -u run_v3.py train "$arm" >"logs/${arm}_train.log" 2>&1
  echo "[$(date -Iseconds)] === TRAIN DONE $arm ==="
  "$PY" -u run_v3.py evaluate "$arm" >"logs/${arm}_eval.log" 2>&1
  echo "[$(date -Iseconds)] === EVAL DONE $arm ==="
  "$PY" -u keep_results_v3.py "$arm" val
  echo "[$(date -Iseconds)] === ARCHIVE VAL DONE $arm ==="
  "$PY" -u diagnose.py --arm "$arm" --phase base >"logs/${arm}_diag.log" 2>&1
  echo "[$(date -Iseconds)] === BASE DIAG DONE $arm ==="
  "$PY" -u keep_results_v3.py "$arm" diagnosis
  echo "[$(date -Iseconds)] === DONE $arm ==="
}

run_arm HistCLS_SchemeA_P2
run_arm HistCLS_CloseMosaic0

echo "[$(date -Iseconds)] === ALL V3 ARMS COMPLETED ==="
