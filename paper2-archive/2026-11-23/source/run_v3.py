"""V3 training & evaluation strictly on original October HistoricalCLS with QuadBest saving:
- best.pt             (COCO AP50:95 best)
- best_ap50.pt        (COCO AP50 best)
- best_native.pt      (native Ultralytics mAP50-95 best)
- best_native_ap50.pt (native Ultralytics mAP50 best)
- last.pt             (fixed E100)
"""
import argparse
import collections
import json
import os
import sys
import time
from pathlib import Path
import torch

STAGE = Path(__file__).resolve().parent
sys.path[:0] = [str(STAGE)]
import run_v2
import candidates_v3
from coco_selection import COCOSelectionTrainer
from ultralytics.utils.torch_utils import unwrap_model

sha, dump, now, proof = run_v2.sha, run_v2.dump, run_v2.now, run_v2.proof
reference = run_v2.reference


def prepare(arm: str):
    if arm not in candidates_v3.ARMS:
        raise ValueError(f"Unknown v3 arm: {arm}")
    base = run_v2.load_historical_base()
    folder = proof(arm)
    folder.mkdir(exist_ok=True, parents=True)
    model, checks = candidates_v3.prepare(base, arm)
    init = folder / "initial.pt"
    if init.exists():
        assert reference.state_sha(reference.load_model(init)) == reference.state_sha(model)
    else:
        torch.save(dict(model=model, ema=None, epoch=-1, optimizer=None), init)
    dump(folder / "candidate_checks.json", checks)

    overrides = dict(candidates_v3.PROTOCOL_OVERRIDES[arm])
    config = dict(json.loads((run_v2.REF / "protocol.json").read_text())["common_protocol"])
    config.update(overrides)
    config.update(
        model=str(init),
        data=str(run_v2.DATA_YAML),
        project=str(STAGE / "runs"),
        name=arm,
        exist_ok=False,
    )
    assert (config["batch"], config["epochs"], config["imgsz"], config["seed"]) == (32, 100, 640, 42)
    assert config["mosaic"] == 1.0
    expected_cm = overrides.get("close_mosaic", 10)
    assert config["close_mosaic"] == expected_cm

    splits = {}
    for name, expected in (("train", 6000), ("val", 1000)):
        arr = (run_v2.REF / "data" / f"{name}.txt").read_text().splitlines()
        assert len(arr) == len(set(arr)) == expected and all("/test/" not in x.lower() for x in arr)
        splits[name] = set(Path(x).name for x in arr)
    assert splits["train"].isdisjoint(splits["val"])

    runtime_files = [
        Path(__file__),
        Path(candidates_v3.__file__),
        Path(run_v2.__file__),
        STAGE / "historical_head.py",
        run_v2.HIST_INIT,
        run_v2.DATA_YAML,
        run_v2.REF / "data/train.txt",
        run_v2.REF / "data/val.txt",
        reference.DATA / "annotations/instances_train.json",
        reference.DATA / "annotations/instances_val.json",
        Path(sys.modules["models"].__file__),
        Path(reference.__file__),
        Path(sys.modules["coco_selection"].__file__),
        Path(sys.modules["rebuilt_model"].__file__),
        Path(sys.modules["pkinet_v2_author"].__file__),
    ]
    contract = dict(
        time=now(),
        arm=arm,
        parent_family="HistoricalCLS (October byte-exact original)",
        parent_initial=str(run_v2.HIST_INIT),
        parent_initial_sha256=run_v2.HIST_INIT_SHA,
        parent_state_digests=run_v2.EXPECTED_HIST,
        initial=str(init),
        initial_sha256=sha(init),
        state_sha=reference.state_sha(model),
        params=sum(p.numel() for p in model.parameters()),
        added_params=checks["added"],
        protocol_overrides=overrides,
        quad_best_saving=True,
        saved_best_rules=[
            "best (COCO AP50:95)",
            "best_ap50 (COCO AP50)",
            "best_native (native mAP50-95)",
            "best_native_ap50 (native mAP50)",
            "last (E100)",
        ],
        hashes={str(p): sha(p) for p in runtime_files},
        protocol=config,
    )
    dump(folder / "contract.json", contract)
    print(
        f"[prepare OK] {arm} (parent=HistoricalCLS {run_v2.OLD_ALL[:10]}): "
        f"params={contract['params']} (+{checks['added']}), overrides={overrides}, "
        f"checks={json.dumps(checks['checks'])}"
    )
    return contract


def train(arm: str):
    if (STAGE / "runs" / arm).exists():
        raise FileExistsError(f"Existing run: {arm}")
    c = prepare(arm)
    folder = proof(arm)
    expected = collections.Counter(Path(x).name for x in (run_v2.REF / "data/train.txt").read_text().splitlines())
    expected_cm = c["protocol"]["close_mosaic"]

    class QuadBestTrainer(COCOSelectionTrainer):
        def get_model(self, cfg=None, weights=None, verbose=True):
            return reference.load_model(c["initial"])

        def preprocess_batch(self, batch):
            self.exposure.update(Path(p).name for p in batch["im_file"])
            result = super().preprocess_batch(batch)
            if not hasattr(self, "first_receipt"):
                self.first_receipt = run_v2.first_batch(result)
                hist_fb = json.loads((proof("HistoricalCLS") / "first_batch.json").read_text())
                for k in ("img", "cls", "bboxes", "batch_idx", "names", "shape"):
                    assert self.first_receipt[k] == hist_fb[k], f"First batch mismatch vs HistoricalCLS on {k}!"
                dump(folder / "first_batch.json", dict(**self.first_receipt, matches_historical_cls=True))
            return result

        def optimizer_step(self):
            before = self.scaler.get_scale()
            super().optimizer_step()
            if self.scaler.get_scale() < before:
                self.skips += 1
            else:
                self.success += 1

        def validate(self):
            metrics, fitness = super().validate()
            # 1) COCO AP50 best
            k50 = (self.validator.coco_main["AP50"], self.validator.coco_main["AP50_95"])
            b50 = getattr(self, "_best_ap50_key", None)
            self._new_ap50_best = b50 is None or k50 > b50
            if self._new_ap50_best:
                self._best_ap50_key = k50
            # 2) Native mAP50-95 best
            kn95 = (metrics["metrics/mAP50-95(B)"], metrics["metrics/mAP50(B)"])
            bn95 = getattr(self, "_best_nat95_key", None)
            self._new_nat95_best = bn95 is None or kn95 > bn95
            if self._new_nat95_best:
                self._best_nat95_key = kn95
            # 3) Native mAP50 best
            kn50 = (metrics["metrics/mAP50(B)"], metrics["metrics/mAP50-95(B)"])
            bn50 = getattr(self, "_best_nat50_key", None)
            self._new_nat50_best = bn50 is None or kn50 > bn50
            if self._new_nat50_best:
                self._best_nat50_key = kn50
            return metrics, fitness

        def save_model(self):
            super().save_model()
            raw_bytes = self.last.read_bytes()
            path = self.save_dir / "checkpoint_epoch_map.json"
            mapping = json.loads(path.read_text()) if path.exists() else {}
            changed = False
            for flag, fname, key in (
                ("_new_ap50_best", "best_ap50.pt", "best_ap50"),
                ("_new_nat95_best", "best_native.pt", "best_native"),
                ("_new_nat50_best", "best_native_ap50.pt", "best_native_ap50"),
            ):
                if getattr(self, flag, False):
                    (self.wdir / fname).write_bytes(raw_bytes)
                    mapping[key] = self.epoch + 1
                    changed = True
            if changed:
                dump(path, mapping)

    def started(t):
        m = unwrap_model(t.model)
        assert reference.state_sha(m) == c["state_sha"]
        assert len(t.train_loader.dataset) == 6000 and len(t.test_loader.dataset) == 1000
        assert sum(len(row["cls"]) for row in t.train_loader.dataset.labels) == 15619
        assert sum(len(row["cls"]) for row in t.test_loader.dataset.labels) == 2612
        assert type(t.optimizer).__name__ == "MuSGD" and t.amp and t.batch_size == 32 and t.epochs == 100
        assert t.args.nbs == 64 and t.accumulate == 2
        assert t.args.close_mosaic == expected_cm and t.args.mosaic == 1.0
        opt_ids = {id(p) for g in t.optimizer.param_groups for p in g["params"]}
        model_ids = {id(p) for p in m.parameters() if p.requires_grad}
        assert model_ids == opt_ids, f"Optimizer param mismatch: missing={len(model_ids - opt_ids)}"
        t.success = t.skips = 0
        criterion = m.init_criterion()
        assert all(
            branch.assigner.stride == [8.0, 16.0, 32.0] and branch.reg_max == 1
            for branch in (criterion.one2many, criterion.one2one)
        )
        m.criterion = criterion
        torch.cuda.reset_peak_memory_stats()
        t.peak_alloc = t.peak_res = 0
        dump(
            folder / "runtime.json",
            dict(
                time=now(),
                pid=os.getpid(),
                gpu=torch.cuda.get_device_name(0),
                parent_family="HistoricalCLS",
                parent_initial_sha256=run_v2.HIST_INIT_SHA,
                state_sha=c["state_sha"],
                params=c["params"],
                added_params=c["added_params"],
                close_mosaic=int(t.args.close_mosaic),
                optimizer=type(t.optimizer).__name__,
                quad_best_saving=True,
                training=True,
            ),
        )

    def epoch_start(t):
        t.exposure = collections.Counter()
        t.epoch_wall = time.monotonic()

    def record(t):
        if not t.validator.training:
            return
        assert t.exposure == expected and unwrap_model(t.model).criterion.updates == t.epoch + 1
        t.peak_alloc = max(t.peak_alloc, torch.cuda.max_memory_allocated())
        t.peak_res = max(t.peak_res, torch.cuda.max_memory_reserved())
        row = dict(
            time=now(),
            epoch=t.epoch + 1,
            primary=sum(t.exposure.values()),
            seconds=time.monotonic() - t.epoch_wall,
            successful_steps=t.success,
            amp_skips=t.skips,
            metrics=t.metrics,
            peak_allocated_GiB=t.peak_alloc / 2**30,
            peak_reserved_GiB=t.peak_res / 2**30,
        )
        with (folder / "epochs.jsonl").open("a") as f:
            f.write(json.dumps(row) + "\n")
        dump(folder / "status.json", row)

    t = QuadBestTrainer(overrides=c["protocol"])
    for event, fn in (("on_train_start", started), ("on_train_epoch_start", epoch_start), ("on_fit_epoch_end", record)):
        t.add_callback(event, fn)
    t.train()
    rows = [json.loads(x) for x in (folder / "epochs.jsonl").read_text().splitlines()]
    assert [x["epoch"] for x in rows] == list(range(1, 101))
    assert all(sha(Path(path)) == h for path, h in c["hashes"].items())
    for fn in ("best.pt", "best_ap50.pt", "best_native.pt", "best_native_ap50.pt", "last.pt"):
        assert (STAGE / "runs" / arm / "weights" / fn).is_file(), f"Missing {fn}"
    dump(
        folder / "trained.json",
        dict(
            time=now(),
            epochs=100,
            primary=sum(x["primary"] for x in rows),
            parent_family="HistoricalCLS",
            state_sha=c["state_sha"],
            close_mosaic=expected_cm,
            quad_best_saving=True,
        ),
    )


def evaluate(arm: str):
    folder = proof(arm)
    assert (folder / "trained.json").is_file()
    result = {}
    epoch_map_path = STAGE / "runs" / arm / "checkpoint_epoch_map.json"
    if epoch_map_path.is_file():
        result["checkpoint_epoch_map"] = json.loads(epoch_map_path.read_text())
    seen_sha = {}
    for kind in ("best", "best_ap50", "best_native", "best_native_ap50", "last"):
        ck = STAGE / "runs" / arm / "weights" / f"{kind}.pt"
        if not ck.is_file():
            continue
        digest = sha(ck)
        path = folder / "evaluation" / kind
        if (path / "metrics.json").exists():
            r = json.loads((path / "metrics.json").read_text())
            assert r["sha256"] == digest
        elif digest in seen_sha:
            r = dict(result[seen_sha[digest]], checkpoint=str(ck), same_as=seen_sha[digest])
            path.mkdir(parents=True, exist_ok=True)
            dump(path / "metrics.json", r)
        else:
            if path.exists():
                path = path.with_name(kind + "_" + str(time.time_ns()))
            r = reference.evaluate_checkpoint(ck, path, run_v2.DATA_YAML)
        seen_sha[digest] = kind
        result[kind] = r
    dump(folder / "evaluation/summary.json", result)
    print(f"[evaluate OK] {arm}:")
    for k in ("best", "best_ap50", "best_native", "best_native_ap50", "last"):
        if k in result:
            m = result[k]["main"]
            ep = result.get("checkpoint_epoch_map", {}).get(k, "?")
            print(
                f"  {k:<16} (E{ep}): AP50:95={m['AP50_95']*100:.3f}%, AP50={m['AP50']*100:.3f}%, "
                f"AP75={m['AP75']*100:.3f}%, S={m['AP_small']*100:.3f}%, M={m['AP_medium']*100:.3f}%, L={m['AP_large']*100:.3f}%"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("prepare", "train", "evaluate"))
    parser.add_argument("arm", choices=candidates_v3.ARMS)
    args = parser.parse_args()
    {"prepare": prepare, "train": train, "evaluate": evaluate}[args.phase](args.arm)
