#!/usr/bin/env bash
set -euo pipefail

WORKDIR="/root/YOLO26N-FCHL/reports/paper2_cls_candidate_diagnosis_640_20261120_42106"
PY="/opt/conda/bin/python"
cd "$WORKDIR"
mkdir -p logs

exec 9>/tmp/paper2_v7_queue.lock
flock -n 9 || { echo "Another paper2_v7_queue is already running!"; exit 1; }

run_arm() {
  local arm="$1"
  local desc="$2"
  echo "[$(date -Iseconds)] === START ${arm} (${desc}) ==="
  "$PY" -u run_v7.py train "$arm" >"logs/${arm}_train.log" 2>&1
  echo "[$(date -Iseconds)] === TRAIN DONE ${arm} ==="
  "$PY" -u run_v7.py evaluate "$arm" >"logs/${arm}_eval.log" 2>&1
  echo "[$(date -Iseconds)] === EVAL DONE ${arm} ==="
  "$PY" -u keep_results_v7.py "$arm" val
  echo "[$(date -Iseconds)] === ARCHIVED ${arm} ==="
}

run_arm "b0"      "Native YOLO26n 640 Uniform"
run_arm "b0_w50"  "Native YOLO26n + #5 256/384"
run_arm "b0_p2"   "B0 + P2 (HistCLS minus PKI2) 640 Uniform"
run_arm "b1_768"  "B1 768 Uniform"
run_arm "b1_w448" "B1 + #5 192/448"

echo "[$(date -Iseconds)] === START ZERO-RETRAIN BENCH & DIAG ==="
"$PY" -u bench_and_diag_v7.py >"logs/v7_bench_and_diag.log" 2>&1
echo "[$(date -Iseconds)] === BENCH & DIAG DONE ==="
"$PY" -u keep_results_v7.py v7 bench
echo "[$(date -Iseconds)] === ALL V7 ABLATION ARMS + ZERO-RETRAIN BENCH COMPLETED ==="
