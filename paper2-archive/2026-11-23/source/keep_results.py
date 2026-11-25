"""Keep the selected weights and their evidence together on the mounted dataset volume."""
import hashlib
import json
import os
import sys
import tarfile
from pathlib import Path

stage = Path(__file__).resolve().parent
arm, phase = sys.argv[1:]
assert arm in {"C0","A","H1_L1","Dual_L1","Dual_L1_static","G1_P2_40","G1_P2_80","NMS075"} and phase in {"val","test","diagnosis"}
records = stage / "records" / arm
dest = Path("/SVRDD/SVRDD_YOLO_READY/_paper2_archives") / stage.name
dest.mkdir(exist_ok=True)
name = f"{arm}_{phase}.tar.gz"
target, checksum = dest / name, dest / (name + ".sha256")
if target.exists() and checksum.exists():
    with target.open("rb") as f:
        assert hashlib.file_digest(f, "sha256").hexdigest() == checksum.read_text().split()[0]
    print(target, flush=True)
    sys.exit(0)
files = {}
def add(base, *paths):
    for path in paths:
        assert path.exists(), path
        for f in sorted(path.rglob("*")) if path.is_dir() else [path]:
            if f.is_file() and "__pycache__" not in f.parts and not f.name.endswith(".tmp"):
                files[str(f.relative_to(base))] = f
if arm == "NMS075":
    assert phase == "val"
    add(stage,records/"evaluation",stage/"decision.json",stage/"nms_control.py")
elif phase == "val":
    run = stage / "runs" / arm
    add(stage, *(stage / n for n in ("run.py","candidates.py","diagnose.py","evaluate_test.py","evaluation_plan.json","keep_results.py")),
        records/"evaluation", records/"epochs.jsonl", records/"first_batch.json", records/"trained.json",
        records/"runtime.json", records/"contract.json", run/"results.csv", run/"args.yaml", run/"checkpoint_epoch_map.json",
        run/"weights/best.pt", run/"weights/last.pt")
    if arm != "C0":
        add(stage, records/"initial.pt", records/"candidate_checks.json", stage/"decision.json")
    else:
        add(stage,stage/"environment_repair.json",stage/"preflight.py",stage/"candidate_gradient_checks.json",
            stage/"choose_from_val.py",stage/"after_diagnosis.sh",stage/"nms_control.py",
            stage/"diagnostic_smoke_trained/receipt.json",stage/"diagnostic_smoke_trained/match_receipt.json")
        a = stage.parent/"paper2_geometry_semantic_640_20261109_42266"
        b = stage.parent/"paper2_cls_p2p3_snake_640_20261115_42264"
        for f in [a/"source", b/"source/dependencies", a/"initialization/Cls_initial_seed42.pt", b/"protocol.json", b/"svrdd7.absolute.yaml", b/"data/train.txt", b/"data/val.txt"]:
            add(stage.parent,f)
elif phase == "test":
    assert arm == "C0"
    add(stage, records/"test",stage/"decision.json",stage/"evaluation_plan.json")
else:
    diagnosis = stage/"diagnostics"/arm
    assert json.loads((diagnosis/"summary.json").read_text())["complete"]
    add(stage, diagnosis/"summary.json", *diagnosis.glob("*/gt.jsonl"), *diagnosis.glob("*/detections_at_025.jsonl"))
tmp = target.with_suffix(f".{os.getpid()}.tmp")
with tarfile.open(tmp,"w:gz",compresslevel=2,dereference=True) as tar:
    for key,f in sorted(files.items()): tar.add(f,arcname=key,recursive=False)
with tmp.open("rb") as f: digest = hashlib.file_digest(f,"sha256").hexdigest()
tmp.replace(target)
checksum.write_text(f"{digest}  {name}\n")
print(target, digest, flush=True)
