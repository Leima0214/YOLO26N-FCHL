#!/usr/bin/env bash
set -euo pipefail

WORKDIR="/root/YOLO26N-FCHL/reports/paper2_cls_candidate_diagnosis_640_20261120_42106"
PY="/opt/conda/bin/python"
cd "$WORKDIR"
mkdir -p logs

exec 9>/tmp/paper2_v6_queue.lock
flock -n 9 || { echo "Another paper2_v6_queue is already running!"; exit 1; }

run_b1() {
  echo "[$(date -Iseconds)] === START b1 ==="
  "$PY" -u run_v6.py train b1 >"logs/b1_train.log" 2>&1
  echo "[$(date -Iseconds)] === TRAIN DONE b1 ==="
  "$PY" -u run_v6.py evaluate b1 >"logs/b1_eval.log" 2>&1
  echo "[$(date -Iseconds)] === EVAL (640+768+slices) DONE b1 ==="
  "$PY" -u keep_results_v6.py b1 val
  echo "[$(date -Iseconds)] === ARCHIVE VAL DONE b1 ==="
  "$PY" -u diagnose.py --arm b1 --phase base >"logs/b1_diag.log" 2>&1
  echo "[$(date -Iseconds)] === BASE DIAG DONE b1 ==="
  "$PY" -u keep_results_v6.py b1 diagnosis
  echo "[$(date -Iseconds)] === DONE b1 ==="
}

run_warp50() {
  echo "[$(date -Iseconds)] === START warp50 ==="
  "$PY" -u run_v6.py train warp50 >"logs/warp50_train.log" 2>&1
  echo "[$(date -Iseconds)] === TRAIN DONE warp50 ==="
  "$PY" -u run_v6.py evaluate warp50 >"logs/warp50_eval.log" 2>&1
  echo "[$(date -Iseconds)] === EVAL (640+768+slices) DONE warp50 ==="
  "$PY" -u keep_results_v6.py warp50 val
  echo "[$(date -Iseconds)] === DONE warp50 ==="
}

run_b1
run_warp50

echo "[$(date -Iseconds)] === ALL V6 ARMS COMPLETED ==="
