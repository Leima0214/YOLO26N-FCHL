#!/usr/bin/env bash
set -euo pipefail
cd /root/YOLO26N-FCHL/reports/paper2_cls_candidate_diagnosis_640_20261120_42106
py=/opt/conda/bin/python
exec 9>logs/v2_queue.lock
flock -n 9 || { echo "v2_queue already running"; exit 0; }

run_arm() {
    local arm="$1"
    local diag_phase="$2"
    echo "[$(date -Iseconds)] === START $arm (parent=HistoricalCLS) ==="
    if [[ ! -f "records/$arm/trained.json" ]]; then
        "$py" -Bu run_v2.py train "$arm" >> "logs/$arm.train.log" 2>&1
    fi
    "$py" -Bu run_v2.py evaluate "$arm" >> "logs/$arm.val.log" 2>&1
    "$py" -Bu keep_results_v2.py "$arm" val || echo "backup $arm val skipped"
    "$py" -Bu diagnose.py --arm "$arm" --phase "$diag_phase" >> "logs/$arm.diagnosis.log" 2>&1
    "$py" -Bu keep_results_v2.py "$arm" diagnosis || echo "backup $arm diagnosis skipped"
    echo "[$(date -Iseconds)] === DONE $arm ==="
}

run_arm "HistCLS_O2M_RegQual" "base"
run_arm "HistCLS_AntiAlias_L01" "all"
echo "[$(date -Iseconds)] === ALL HistoricalCLS V2 QUEUE ARMS COMPLETE ==="
