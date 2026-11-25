#!/usr/bin/env bash
set -euo pipefail
cd /root/YOLO26N-FCHL/reports/paper2_cls_candidate_diagnosis_640_20261120_42106
py=/opt/conda/bin/python
exec 9>logs/hist_dualbest.lock
flock -n 9 || { echo "hist_dualbest already running"; exit 0; }

ARM="HistoricalCLS_DualBest"
echo "[$(date -Iseconds)] === START $ARM (pure original October HistoricalCLS, quad-best saving) ==="
if [[ ! -f "records/$ARM/trained.json" ]]; then
    "$py" -Bu run_hist_dualbest.py train >> "logs/$ARM.train.log" 2>&1
fi
"$py" -Bu run_hist_dualbest.py evaluate >> "logs/$ARM.val.log" 2>&1
# Archive all 5 checkpoints (best, best_ap50, best_native, best_native_ap50, last) + records to /SVRDD
"$py" - <<'PY'
import hashlib, os, tarfile
from pathlib import Path
stage = Path('/root/YOLO26N-FCHL/reports/paper2_cls_candidate_diagnosis_640_20261120_42106')
arm = 'HistoricalCLS_DualBest'
dest = Path('/SVRDD/SVRDD_YOLO_READY/_paper2_archives') / stage.name
dest.mkdir(exist_ok=True, parents=True)
target = dest / f'{arm}_val.tar.gz'
checksum = dest / f'{arm}_val.tar.gz.sha256'
files = {}
for p in [
    stage/'run_hist_dualbest.py', stage/'run_v2.py', stage/'historical_head.py',
    stage/'records'/arm,
    stage/'runs'/arm/'results.csv', stage/'runs'/arm/'args.yaml', stage/'runs'/arm/'checkpoint_epoch_map.json',
    stage/'runs'/arm/'weights/best.pt', stage/'runs'/arm/'weights/best_ap50.pt',
    stage/'runs'/arm/'weights/best_native.pt', stage/'runs'/arm/'weights/best_native_ap50.pt',
    stage/'runs'/arm/'weights/last.pt',
]:
    if not p.exists(): continue
    for f in (sorted(p.rglob('*')) if p.is_dir() else [p]):
        if f.is_file() and '__pycache__' not in f.parts and not f.name.endswith('.tmp'):
            files[str(f.relative_to(stage))] = f
tmp = target.with_suffix(f'.{os.getpid()}.tmp')
with tarfile.open(tmp, 'w:gz', compresslevel=2, dereference=True) as tar:
    for k, f in sorted(files.items()):
        tar.add(f, arcname=k, recursive=False)
with tmp.open('rb') as f:
    digest = hashlib.file_digest(f, 'sha256').hexdigest()
tmp.replace(target)
checksum.write_text(f'{digest}  {target.name}\n')
print(target, digest, flush=True)
PY
echo "[$(date -Iseconds)] === DONE $ARM ==="
