"""Authorized C/A/R screening queue: preflight, independent smoke, fresh 30E and paired Val evidence."""

import csv
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/operator_screen_30E_seed42_20261005"
DATA = ROOT / "configs/datasets/japan4.local.yaml"
BASELINE_NAME = "B0_O2M_30E_seed42_20261004"
BASELINE = ROOT / "runs/paper2" / BASELINE_NAME
RUNS = {tag: f"{tag}_O2M_30E_seed42_20261005" for tag in "CAR"}


def prepare_morphology_subset():
    """Blind sampling by class/size, never by model success. Labels remain for human review."""
    rows = list(csv.DictReader((OUT / "baseline_diagnostic/per_gt_candidates.csv").open()))
    rng = random.Random(42)
    selected = []
    for cls in ("D00", "D10", "D20", "D40"):
        groups = {size: [r for r in rows if r["class"] == cls and r["size"] == size]
                  for size in ("small", "medium", "large")}
        for group in groups.values():
            rng.shuffle(group)
        subset = []
        while len(subset) < 50 and any(groups.values()):
            for group in groups.values():
                if group and len(subset) < 50:
                    subset.append(group.pop())
        selected.extend(subset)
    fields = ["image", "gt_index", "class", "size", "input_area", "input_short_side", "aspect_ratio"]
    fields += ["directionality", "structure", "visibility", "uncertain", "notes"]
    with (OUT / "morphology_subset.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in fields} for row in selected)
    (OUT / "morphology_subset_README.txt").write_text(
        "200 GT sampled by class and input-size with seed42, independent of performance.\n"
        "Open /Japan4-V3/Japan4-cleanV3/images/val/<image> and labels/val/<stem>.txt; gt_index is zero-based label line.\n"
        "Human labels are intentionally blank: directionality=clear/multiple/none/uncertain; "
        "structure=linear/network/region/mixed/uncertain; visibility=clear/weak/uncertain.\n"
        "Class ID and HBB aspect ratio are not morphology ground truth. No manual annotation claimed.\n")
    return len(selected)


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    status = {"pid": os.getpid(), "state": "starting", "stages": [],
              "test_sealed": True, "automatic_monitoring": False}

    def save():
        temp = OUT / "status.tmp"
        temp.write_text(json.dumps(status, indent=2))
        temp.replace(OUT / "status.json")

    def run(stage, args):
        record = {"stage": stage, "command": [sys.executable, "-u", *map(str, args)], "started_at": time.time()}
        status["stages"].append(record)
        status.update(state="running", stage=stage)
        save()
        with (OUT / f"{stage}.log").open("w") as log:
            child = subprocess.Popen(record["command"], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            status["child_pid"] = child.pid
            save()
            record["exit_code"] = child.wait()
        record["finished_at"] = time.time()
        save()
        if record["exit_code"]:
            raise RuntimeError(f"{stage} failed: exit {record['exit_code']}")

    common = ["--data", DATA, "--weights", ROOT / "weights/yolo26n.pt", "--imgsz", "640",
              "--batch", "32", "--workers", "8", "--seed", "42", "--device", "0"]
    try:
        # These commands write only this job's artifacts; Test is never loaded.
        run("dataset_audit", ["scripts/audit_japan4_dataset.py", "--data", DATA, "--output", OUT / "dataset_audit.json"])
        run("migration_check", ["tests/check_migration.py"])
        run("operator_engineering", ["tests/check_paper2_operators.py", "--weights", ROOT / "weights/yolo26n.pt",
                                      "--device", "cuda:0", "--report", OUT / "engineering/results.json"])
        run("baseline_diagnostic", ["scripts/diagnose_japan4_head_candidates.py", "--model", f"B0={BASELINE}/weights/best.pt",
                                    "--data", DATA, "--output", OUT / "baseline_diagnostic", "--branches", "o2m",
                                    "--skip-val-sweep", "--workers", "8"])
        status["human_subset_GT_count"] = prepare_morphology_subset()
        status["human_subset_annotated"] = False
        reference = json.loads((ROOT / "runtime_meta" / BASELINE_NAME / "effective_optimizer.json").read_text())
        reference_args = json.loads((ROOT / "runtime_meta" / BASELINE_NAME / "resolved_arguments.json").read_text())
        for tag in "CAR":
            smoke = f"T2_{tag}_O2M_1E_seed42_20261005"
            run(f"smoke_{tag}", ["scripts/train_paper2.py", "--candidate", f"{tag.lower()}_o2m",
                                  "--name", smoke, "--epochs", "1", *common])
            meta = ROOT / "runtime_meta" / smoke
            actual = json.loads((meta / "effective_optimizer.json").read_text())
            resolved = json.loads((meta / "resolved_arguments.json").read_text())
            assert actual["initial_o2m_state_sha256"] == reference["initial_o2m_state_sha256"], tag
            assert actual["groups"] == reference["groups"], tag
            assert resolved["input_weights_sha256"] == reference_args["input_weights_sha256"], tag
            assert not actual["end2end"] and actual["criterion"] == "v8DetectionLoss", tag
        status["preflight_passed"] = True
        status["initial_shared_state_sha256"] = reference["initial_o2m_state_sha256"]
        save()
        for tag in "CAR":
            run(f"train_{tag}", ["scripts/train_paper2.py", "--candidate", f"{tag.lower()}_o2m",
                                  "--name", RUNS[tag], "--epochs", "30", *common])
            actual = json.loads((ROOT / "runtime_meta" / RUNS[tag] / "effective_optimizer.json").read_text())
            assert actual["initial_o2m_state_sha256"] == status["initial_shared_state_sha256"], tag
        checkpoints = {"B0": BASELINE / "weights/best.pt",
                       **{tag: ROOT / "runs/paper2" / name / "weights/best.pt" for tag, name in RUNS.items()}}
        selections = [item for label, path in checkpoints.items() for item in ("--checkpoint", f"{label}={path}")]
        run("unified_evaluation", ["scripts/eval_paper2.py", *selections, "--data", DATA, "--output", OUT / "evaluation"])
        run_args = [item for label, path in {"B0": BASELINE, **{tag: ROOT / "runs/paper2" / name for tag, name in RUNS.items()}}.items()
                    for item in ("--run", f"{label}={path}", "--eval", f"{label}={OUT}/evaluation")]
        run("comparison", ["scripts/compare_japan4_runs.py", *run_args, "--output", OUT / "comparison"])
        candidates = [item for label, path in checkpoints.items() for item in ("--model", f"{label}={path}")]
        run("paired_diagnostic", ["scripts/diagnose_japan4_head_candidates.py", *candidates, "--data", DATA,
                                  "--output", OUT / "paired_diagnostic", "--branches", "o2m", "--reference", "B0",
                                  "--skip-val-sweep", "--workers", "8"])
        status.update(state="completed", finished_at=time.time())
        save()
    except BaseException as error:
        status.update(state="failed", error=repr(error), finished_at=time.time())
        save()
        raise


if __name__ == "__main__":
    main()
