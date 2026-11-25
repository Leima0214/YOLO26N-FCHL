"""Archive completed official YOLO horizontal arm weights & evidence onto /SVRDD and prune epoch*.pt snapshots."""
import hashlib
import os
import sys
import tarfile
from pathlib import Path

stage = Path(__file__).resolve().parent
arm = sys.argv[1]
dest = Path("/SVRDD/SVRDD_YOLO_READY/_paper2_archives") / stage.name
dest.mkdir(exist_ok=True, parents=True)

files = {}


def add(base, *paths):
    for path in paths:
        if not path.exists():
            continue
        for f in sorted(path.rglob("*")) if path.is_dir() else [path]:
            if f.is_file() and "__pycache__" not in f.parts and not f.name.endswith(".tmp"):
                files[str(f.relative_to(base))] = f


records = stage / "records" / arm
run = stage / "runs" / arm
add(
    stage,
    stage / "run_official_yolo.py",
    stage / "preflight_official_yolo.py",
    stage / "keep_official_yolo.py",
    stage / "run_v2.py",
    stage / "run_v7.py",
    stage / "diagnostics/preflight_official_yolos.json",
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

name = f"{arm}_val.tar.gz"
target, checksum = dest / name, dest / (name + ".sha256")
tmp = target.with_suffix(f".{os.getpid()}.tmp")
with tarfile.open(tmp, "w:gz", compresslevel=2, dereference=True) as tar:
    for key, f in sorted(files.items()):
        tar.add(f, arcname=key, recursive=False)
with tmp.open("rb") as f:
    digest = hashlib.file_digest(f, "sha256").hexdigest()
tmp.replace(target)
checksum.write_text(f"{digest}  {name}\n")

# Clean up intermediate periodic snapshots after archive verification so / overlay remains spacious.
removed = 0
wdir = run / "weights"
if wdir.is_dir():
    for snap in wdir.glob("epoch*.pt"):
        snap.unlink()
        removed += 1

print(f"{target} {digest} (removed_epoch_snapshots={removed})", flush=True)
