#!/usr/bin/env bash
set -euo pipefail

WORKDIR="/root/YOLO26N-FCHL/reports/paper2_cls_candidate_diagnosis_640_20261120_42106"
PY="/opt/conda/bin/python"
cd "$WORKDIR"
mkdir -p logs

exec 9>/tmp/paper2_official_yolo_queue.lock
flock -n 9 || { echo "Another paper2_official_yolo_queue is already running!"; exit 1; }

run_arm() {
  local arm="$1"
  local desc="$2"
  if [ -s "records/${arm}/evaluation/summary.json" ]; then
    echo "[$(date -Iseconds)] === SKIP COMPLETED ${arm} ==="
    return 0
  fi
  echo "[$(date -Iseconds)] === START ${arm} (${desc}) ==="
  "$PY" -u run_official_yolo.py train "$arm" >"logs/${arm}_train.log" 2>&1
  echo "[$(date -Iseconds)] === TRAIN DONE ${arm} ==="
  "$PY" -u run_official_yolo.py evaluate "$arm" >"logs/${arm}_eval.log" 2>&1
  echo "[$(date -Iseconds)] === EVAL DONE ${arm} ==="
  "$PY" -u keep_official_yolo.py "$arm"
  echo "[$(date -Iseconds)] === ARCHIVED ${arm} ==="
}

run_arm "v8n"  "YOLOv8n (640, AdamW, 100E)"
run_arm "v9t"  "YOLOv9t (640, AdamW, 100E)"
run_arm "v10n" "YOLOv10n (640, AdamW, 100E, E2E)"
run_arm "y11n" "YOLO11n (640, AdamW, 100E)"
run_arm "y12n" "YOLO12n (640, AdamW, 100E)"
run_arm "y26s" "YOLO26s (640, MuSGD, 100E, O2M+O2O)"

echo "[$(date -Iseconds)] === ALL 6 OFFICIAL YOLO BASELINES COMPLETED ==="
