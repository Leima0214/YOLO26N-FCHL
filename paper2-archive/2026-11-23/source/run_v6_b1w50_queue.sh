#!/usr/bin/env bash
set -euo pipefail

WORKDIR="/root/YOLO26N-FCHL/reports/paper2_cls_candidate_diagnosis_640_20261120_42106"
PY="/opt/conda/bin/python"
cd "$WORKDIR"
mkdir -p logs

exec 9>/tmp/paper2_v6_queue.lock
flock -n 9 || { echo "Another paper2_v6_queue is already running!"; exit 1; }

run_b1_w50() {
  echo "[$(date -Iseconds)] === START b1_w50 (B1 + #5 256/384) ==="
  "$PY" -u run_v6.py train b1_w50 >"logs/b1_w50_train.log" 2>&1
  echo "[$(date -Iseconds)] === TRAIN DONE b1_w50 ==="
  "$PY" -u run_v6.py evaluate b1_w50 >"logs/b1_w50_eval.log" 2>&1
  echo "[$(date -Iseconds)] === EVAL (640+768+slices) DONE b1_w50 ==="
  "$PY" -u keep_results_v6.py b1_w50 val
  echo "[$(date -Iseconds)] === DONE b1_w50 ==="
}

run_p2a_w50() {
  echo "[$(date -Iseconds)] === START p2a_w50 (CLS-P2A + #5 256/384) ==="
  "$PY" -u run_v6.py train p2a_w50 >"logs/p2a_w50_train.log" 2>&1
  echo "[$(date -Iseconds)] === TRAIN DONE p2a_w50 ==="
  "$PY" -u run_v6.py evaluate p2a_w50 >"logs/p2a_w50_eval.log" 2>&1
  echo "[$(date -Iseconds)] === EVAL (640+768+slices) DONE p2a_w50 ==="
  "$PY" -u keep_results_v6.py p2a_w50 val
  echo "[$(date -Iseconds)] === DONE p2a_w50 ==="
}

run_b1_w50
run_p2a_w50

echo "[$(date -Iseconds)] === ALL V6.1 ARMS COMPLETED ==="
