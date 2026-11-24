"""Pure original October HistoricalCLS 100E rerun saving ALL best checkpoints:
- best.pt            (COCO AP50:95 best)
- best_ap50.pt       (COCO AP50 best)
- best_native.pt     (native Ultralytics mAP50-95 best)
- best_native_ap50.pt(native Ultralytics mAP50 best)
- last.pt            (fixed E100)
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
from coco_selection import COCOSelectionTrainer
from ultralytics.utils.torch_utils import unwrap_model

ARM = "HistoricalCLS_DualBest"
sha, dump, now = run_v2.sha, run_v2.dump, run_v2.now


def prepare():
    base = run_v2.load_historical_base()
    folder = run_v2.proof(ARM)
    folder.mkdir(exist_ok=True, parents=True)
    init = folder / "initial.pt"
    if init.exists():
        assert run_v2.reference.state_sha(run_v2.reference.load_model(init)) == run_v2.reference.state_sha(base)
    else:
        torch.save(dict(model=base, ema=None, epoch=-1, optimizer=None), init)
    config = dict(
        json.loads((run_v2.REF / "protocol.json").read_text())["common_protocol"],
        model=str(init),
        data=str(run_v2.DATA_YAML),
        project=str(STAGE / "runs"),
        name=ARM,
        exist_ok=False,
    )
    assert (config["batch"], config["epochs"], config["imgsz"], config["seed"]) == (32, 100, 640, 42)
    contract = dict(
        time=now(),
        arm=ARM,
        parent_family="HistoricalCLS (pure unmodified October byte-exact original)",
        parent_initial=str(run_v2.HIST_INIT),
        parent_initial_sha256=run_v2.HIST_INIT_SHA,
        parent_state_digests=run_v2.EXPECTED_HIST,
        initial=str(init),
        initial_sha256=sha(init),
        state_sha=run_v2.reference.state_sha(base),
        params=sum(p.numel() for p in base.parameters()),
        saved_best_rules=["best (COCO AP50:95)", "best_ap50 (COCO AP50)", "best_native (native mAP50-95)", "best_native_ap50 (native mAP50)", "last (E100)"],
        protocol=config,
    )
    dump(folder / "contract.json", contract)
    print(f"[prepare OK] {ARM}: pure HistoricalCLS {run_v2.OLD_ALL[:10]}, params={contract['params']}")
    return contract


def train():
    if (STAGE / "runs" / ARM).exists():
        raise FileExistsError(f"Existing run: {ARM}")
    c = prepare()
    folder = run_v2.proof(ARM)
    expected = collections.Counter(Path(x).name for x in (run_v2.REF / "data/train.txt").read_text().splitlines())

    class QuadBestTrainer(COCOSelectionTrainer):
        def get_model(self, cfg=None, weights=None, verbose=True):
            return run_v2.reference.load_model(c["initial"])

        def preprocess_batch(self, batch):
            self.exposure.update(Path(p).name for p in batch["im_file"])
            result = super().preprocess_batch(batch)
            if not hasattr(self, "first_receipt"):
                self.first_receipt = run_v2.first_batch(result)
                hist_fb = json.loads((run_v2.proof("HistoricalCLS") / "first_batch.json").read_text())
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
        assert run_v2.digests(m) == run_v2.EXPECTED_HIST
        assert sum(p.numel() for p in m.parameters()) == 3195674
        assert len(t.train_loader.dataset) == 6000 and len(t.test_loader.dataset) == 1000
        assert type(t.optimizer).__name__ == "MuSGD" and t.amp and t.batch_size == 32 and t.epochs == 100
        assert t.args.nbs == 64 and t.accumulate == 2
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
                parent_family="HistoricalCLS (pure original)",
                parent_initial_sha256=run_v2.HIST_INIT_SHA,
                state_sha=c["state_sha"],
                params=3195674,
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
    for fn in ("best.pt", "best_ap50.pt", "best_native.pt", "best_native_ap50.pt", "last.pt"):
        assert (STAGE / "runs" / ARM / "weights" / fn).is_file(), f"Missing {fn}"
    dump(
        folder / "trained.json",
        dict(
            time=now(),
            epochs=100,
            primary=sum(x["primary"] for x in rows),
            parent_family="HistoricalCLS (pure original)",
            state_sha=c["state_sha"],
            quad_best_saving=True,
        ),
    )


def evaluate():
    folder = run_v2.proof(ARM)
    assert (folder / "trained.json").is_file()
    result = {}
    epoch_map_path = STAGE / "runs" / ARM / "checkpoint_epoch_map.json"
    if epoch_map_path.is_file():
        result["checkpoint_epoch_map"] = json.loads(epoch_map_path.read_text())
    seen_sha = {}
    for kind in ("best", "best_ap50", "best_native", "best_native_ap50", "last"):
        ck = STAGE / "runs" / ARM / "weights" / f"{kind}.pt"
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
            r = run_v2.reference.evaluate_checkpoint(ck, path, run_v2.DATA_YAML)
        seen_sha[digest] = kind
        result[kind] = r
    dump(folder / "evaluation/summary.json", result)
    print(f"[evaluate OK] {ARM}:")
    for k in ("best", "best_ap50", "best_native", "best_native_ap50", "last"):
        if k in result:
            m = result[k]["main"]
            ep = result.get("checkpoint_epoch_map", {}).get(k, "?")
            print(
                f"  {k:<16} (E{ep}): AP50:95={m['AP50_95']*100:.3f}%, AP50={m['AP50']*100:.3f}%, "
                f"AP75={m['AP75']*100:.3f}%, S={m['AP_small']*100:.3f}%, M={m['AP_medium']*100:.3f}%, L={m['AP_large']*100:.3f}%"
            )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("phase", choices=("prepare", "train", "evaluate"))
    args = p.parse_args()
    {"prepare": prepare, "train": train, "evaluate": evaluate}[args.phase]()
