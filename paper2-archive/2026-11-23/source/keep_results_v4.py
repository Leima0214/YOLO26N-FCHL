"""Archive V4 arm weights (all 5 checkpoints) and evidence (640 + 768) on /SVRDD."""
import hashlib
import json
import os
import sys
import tarfile
from pathlib import Path

stage = Path(__file__).resolve().parent
arm, phase = sys.argv[1:]
assert phase in {"val", "diagnosis"}
records = stage / "records" / arm
dest = Path("/SVRDD/SVRDD_YOLO_READY/_paper2_archives") / stage.name
dest.mkdir(exist_ok=True, parents=True)
name = f"{arm}_{phase}.tar.gz"
target, checksum = dest / name, dest / (name + ".sha256")
files = {}


def add(base, *paths):
    for path in paths:
        if not path.exists():
            continue
        for f in sorted(path.rglob("*")) if path.is_dir() else [path]:
            if f.is_file() and "__pycache__" not in f.parts and not f.name.endswith(".tmp"):
                files[str(f.relative_to(base))] = f


if phase == "val":
    run = stage / "runs" / arm
    add(
        stage,
        stage / "run_v4.py",
        stage / "candidates_v4.py",
        stage / "candidates_v3.py",
        stage / "run_v2.py",
        stage / "historical_head.py",
        stage / "recover_historical_initial.py",
        stage / "keep_results_v4.py",
        stage / "diagnose.py",
        records / "evaluation",
        records / "epochs.jsonl",
        records / "first_batch.json",
        records / "trained.json",
        records / "runtime.json",
        records / "contract.json",
        records / "initial.pt",
        records / "candidate_checks.json",
        run / "results.csv",
        run / "args.yaml",
        run / "checkpoint_epoch_map.json",
        run / "weights/best.pt",
        run / "weights/best_ap50.pt",
        run / "weights/best_native.pt",
        run / "weights/best_native_ap50.pt",
        run / "weights/last.pt",
    )
else:
    diagnosis = stage / "diagnostics" / arm
    assert json.loads((diagnosis / "summary.json").read_text())["complete"]
    add(stage, diagnosis / "summary.json", *diagnosis.glob("*/gt.jsonl"), *diagnosis.glob("*/detections_at_025.jsonl"))

tmp = target.with_suffix(f".{os.getpid()}.tmp")
with tarfile.open(tmp, "w:gz", compresslevel=2, dereference=True) as tar:
    for key, f in sorted(files.items()):
        tar.add(f, arcname=key, recursive=False)
with tmp.open("rb") as f:
    digest = hashlib.file_digest(f, "sha256").hexdigest()
tmp.replace(target)
checksum.write_text(f"{digest}  {name}\n")
print(target, digest, flush=True)
