#!/usr/bin/env bash
set -euo pipefail

WORKDIR="/root/YOLO26N-FCHL/reports/paper2_cls_candidate_diagnosis_640_20261120_42106"
PY="/opt/conda/bin/python"
cd "$WORKDIR"
mkdir -p logs

exec 9>/tmp/paper2_v5_queue.lock
flock -n 9 || { echo "Another paper2_v5_queue is already running!"; exit 1; }

run_arm() {
  local arm="$1"
  echo "[$(date -Iseconds)] === START $arm ==="
  "$PY" -u run_v5.py train "$arm" >"logs/${arm}_train.log" 2>&1
  echo "[$(date -Iseconds)] === TRAIN DONE $arm ==="
  "$PY" -u run_v5.py evaluate "$arm" >"logs/${arm}_eval.log" 2>&1
  echo "[$(date -Iseconds)] === EVAL (640+768) DONE $arm ==="
  "$PY" -u keep_results_v5.py "$arm" val
  echo "[$(date -Iseconds)] === ARCHIVE VAL DONE $arm ==="
  "$PY" -u diagnose.py --arm "$arm" --phase base >"logs/${arm}_diag.log" 2>&1
  echo "[$(date -Iseconds)] === BASE DIAG DONE $arm ==="
  "$PY" -u keep_results_v5.py "$arm" diagnosis
  echo "[$(date -Iseconds)] === DONE $arm ==="
}

run_arm p2_det
run_arm p2_cm0

echo "[$(date -Iseconds)] === ALL V5 ARMS COMPLETED ==="
