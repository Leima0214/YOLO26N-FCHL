"""V2 training & evaluation strictly on the original October HistoricalCLS parent, with dual best saving (best.pt + best_ap50.pt)."""
import argparse
import collections
import hashlib
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

STAGE = Path(__file__).resolve().parent
ROOT = Path("/root/YOLO26N-FCHL")
REPORTS = ROOT / "reports"
GEO = REPORTS / "paper2_geometry_semantic_640_20261109_42266"
REF = REPORTS / "paper2_cls_p2p3_snake_640_20261115_42264"
HIST_INIT = STAGE / "identity_audit/historical_replay/CLS640_historical_initial.pt"
HIST_INIT_SHA = "71da1a0dd162ec45903f6e146c5852b68c70cfa52bd596de667ffeabae4bfc29"
OLD_PIXELS = "92208b9a827c4a23397785d8a8254f820e456d1abefecbfa4a21c5d85a5eeb43"
DATA_YAML = REF / "svrdd7.absolute.yaml"
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("WANDB_MODE", "disabled")
sys.path[:0] = [
    str(STAGE),
    str(GEO / "source"),
    str(REF / "source/dependencies"),
    str(GEO / "source/dependencies"),
    str(ROOT),
]
import torch
import models  # Pickle identity
import experiment as reference
import historical_head
from recover_historical_initial import OLD_ALL, OLD_P2, OLD_PKI, digests
import candidates_v2
from coco_selection import COCOSelectionTrainer
from ultralytics.utils.torch_utils import unwrap_model

sha, dump = reference.file_sha, reference.dump
EXPECTED_HIST = dict(all=OLD_ALL, P2=OLD_P2, PKI=OLD_PKI)
torch.set_num_threads(2)


def now():
    return datetime.now(ZoneInfo("America/Los_Angeles")).isoformat()


def proof(arm):
    return STAGE / "records" / arm


def load_historical_base():
    assert sha(HIST_INIT) == HIST_INIT_SHA, f"Historical init SHA mismatch: {sha(HIST_INIT)}"
    base = historical_head.bind(reference.load_model(HIST_INIT))
    assert digests(base) == EXPECTED_HIST, f"Historical digest mismatch: {digests(base)}"
    assert sum(p.numel() for p in base.parameters()) == 3195674
    head = base.model[-1]
    bn = [(x.eps, x.momentum) for x in head.p2_fusion.modules() if isinstance(x, torch.nn.BatchNorm2d)]
    assert bn == [(0.001, 0.03)] * 4, f"Historical p2_fusion BN mismatch: {bn}"
    assert head.stride.tolist() == [4.0, 8.0, 16.0, 32.0] and [len(s) for s in head.paper2_pki2.stages] == [2, 2]
    assert base.model[0].conv.stride == (2, 2) and base.model[1].conv.stride == (2, 2)
    return base


def prepare(arm):
    base = load_historical_base()
    folder = proof(arm)
    folder.mkdir(exist_ok=True, parents=True)
    model, checks = candidates_v2.prepare(base, arm)
    init = folder / "initial.pt"
    if init.exists():
        assert reference.state_sha(reference.load_model(init)) == reference.state_sha(model)
    else:
        torch.save(dict(model=model, ema=None, epoch=-1, optimizer=None), init)
    dump(folder / "candidate_checks.json", checks)

    config = dict(
        json.loads((REF / "protocol.json").read_text())["common_protocol"],
        model=str(init),
        data=str(DATA_YAML),
        project=str(STAGE / "runs"),
        name=arm,
        exist_ok=False,
    )
    assert (config["batch"], config["epochs"], config["imgsz"], config["seed"]) == (32, 100, 640, 42)
    splits = {}
    for name, expected in (("train", 6000), ("val", 1000)):
        arr = (REF / "data" / f"{name}.txt").read_text().splitlines()
        assert len(arr) == len(set(arr)) == expected and all("/test/" not in x.lower() for x in arr)
        splits[name] = set(Path(x).name for x in arr)
    assert splits["train"].isdisjoint(splits["val"])

    runtime_files = [
        Path(__file__),
        Path(candidates_v2.__file__),
        STAGE / "historical_head.py",
        HIST_INIT,
        DATA_YAML,
        REF / "data/train.txt",
        REF / "data/val.txt",
        reference.DATA / "annotations/instances_train.json",
        reference.DATA / "annotations/instances_val.json",
        Path(models.__file__),
        Path(reference.__file__),
        Path(sys.modules["coco_selection"].__file__),
        Path(sys.modules["rebuilt_model"].__file__),
        Path(sys.modules["pkinet_v2_author"].__file__),
    ]
    contract = dict(
        time=now(),
        arm=arm,
        parent_family="HistoricalCLS (October byte-exact original)",
        parent_initial=str(HIST_INIT),
        parent_initial_sha256=HIST_INIT_SHA,
        parent_state_digests=EXPECTED_HIST,
        initial=str(init),
        initial_sha256=sha(init),
        state_sha=reference.state_sha(model),
        params=sum(p.numel() for p in model.parameters()),
        dual_best_saving=True,
        hashes={str(p): sha(p) for p in runtime_files},
        protocol=config,
    )
    dump(folder / "contract.json", contract)
    print(f"[prepare OK] {arm} (parent=HistoricalCLS {OLD_ALL[:10]}): params={contract['params']} (+{checks['added']}), checks={json.dumps(checks['checks'])}")
    return contract


def first_batch(batch):
    receipt = {
        k: hashlib.sha256(batch[k].detach().cpu().contiguous().numpy().tobytes()).hexdigest()
        for k in ("img", "cls", "bboxes", "batch_idx")
    }
    receipt.update(names=[Path(p).name for p in batch["im_file"]], shape=list(batch["img"].shape))
    return receipt


def train(arm):
    if (STAGE / "runs" / arm).exists():
        raise FileExistsError(f"Existing run: {arm}")
    c = prepare(arm)
    folder = proof(arm)
    expected = collections.Counter(Path(x).name for x in (REF / "data/train.txt").read_text().splitlines())

    class DualBestTrainer(COCOSelectionTrainer):
        def get_model(self, cfg=None, weights=None, verbose=True):
            return reference.load_model(c["initial"])

        def preprocess_batch(self, batch):
            self.exposure.update(Path(p).name for p in batch["im_file"])
            result = super().preprocess_batch(batch)
            if not hasattr(self, "first_receipt"):
                self.first_receipt = first_batch(result)
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
            key50 = (self.validator.coco_main["AP50"], self.validator.coco_main["AP50_95"])
            best50 = getattr(self, "_best_ap50_key", None)
            self._new_ap50_best = best50 is None or key50 > best50
            if self._new_ap50_best:
                self._best_ap50_key = key50
            return metrics, fitness

        def save_model(self):
            super().save_model()
            if getattr(self, "_new_ap50_best", False):
                (self.wdir / "best_ap50.pt").write_bytes(self.last.read_bytes())
                path = self.save_dir / "checkpoint_epoch_map.json"
                mapping = json.loads(path.read_text()) if path.exists() else {}
                mapping["best_ap50"] = self.epoch + 1
                dump(path, mapping)

    def started(t):
        m = unwrap_model(t.model)
        assert reference.state_sha(m) == c["state_sha"]
        assert len(t.train_loader.dataset) == 6000 and len(t.test_loader.dataset) == 1000
        assert sum(len(row["cls"]) for row in t.train_loader.dataset.labels) == 15619
        assert sum(len(row["cls"]) for row in t.test_loader.dataset.labels) == 2612
        assert type(t.optimizer).__name__ == "MuSGD" and t.amp and t.batch_size == 32 and t.epochs == 100
        assert t.args.nbs == 64 and t.accumulate == 2
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
                parent_initial_sha256=HIST_INIT_SHA,
                state_sha=c["state_sha"],
                params=c["params"],
                optimizer=type(t.optimizer).__name__,
                dual_best_saving=True,
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

    t = DualBestTrainer(overrides=c["protocol"])
    for event, fn in (("on_train_start", started), ("on_train_epoch_start", epoch_start), ("on_fit_epoch_end", record)):
        t.add_callback(event, fn)
    t.train()
    rows = [json.loads(x) for x in (folder / "epochs.jsonl").read_text().splitlines()]
    assert [x["epoch"] for x in rows] == list(range(1, 101))
    assert all(sha(Path(path)) == h for path, h in c["hashes"].items())
    assert (STAGE / "runs" / arm / "weights/best_ap50.pt").is_file(), "best_ap50.pt missing!"
    dump(
        folder / "trained.json",
        dict(
            time=now(),
            epochs=100,
            primary=sum(x["primary"] for x in rows),
            parent_family="HistoricalCLS",
            state_sha=c["state_sha"],
            dual_best_saving=True,
        ),
    )


def evaluate(arm):
    folder = proof(arm)
    assert (folder / "trained.json").is_file()
    result = {}
    epoch_map_path = STAGE / "runs" / arm / "checkpoint_epoch_map.json"
    if epoch_map_path.is_file():
        result["checkpoint_epoch_map"] = json.loads(epoch_map_path.read_text())
    for kind in ["best", "best_ap50", "last"]:
        ck = STAGE / "runs" / arm / "weights" / f"{kind}.pt"
        if not ck.is_file():
            if kind == "best_ap50":
                continue
            raise FileNotFoundError(ck)
        path = folder / "evaluation" / kind
        if (path / "metrics.json").exists():
            r = json.loads((path / "metrics.json").read_text())
            assert r["sha256"] == sha(ck)
        elif kind == "best_ap50" and "best" in result and sha(ck) == result["best"]["sha256"]:
            r = dict(result["best"], checkpoint=str(ck), same_as_best=True)
            path.mkdir(parents=True, exist_ok=True)
            dump(path / "metrics.json", r)
        else:
            if path.exists():
                path = path.with_name(kind + "_" + str(time.time_ns()))
            r = reference.evaluate_checkpoint(ck, path, DATA_YAML)
        result[kind] = r
    dump(folder / "evaluation/summary.json", result)
    print(f"[evaluate OK] {arm}:")
    for k in ("best", "best_ap50", "last"):
        if k in result:
            m = result[k]["main"]
            ep = result.get("checkpoint_epoch_map", {}).get(k, "?")
            print(
                f"  {k:<9} (E{ep}): AP50:95={m['AP50_95']*100:.3f}%, AP50={m['AP50']*100:.3f}%, "
                f"AP75={m['AP75']*100:.3f}%, S={m['AP_small']*100:.3f}%, M={m['AP_medium']*100:.3f}%, L={m['AP_large']*100:.3f}%"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("train", "evaluate", "prepare"))
    parser.add_argument("arm")
    args = parser.parse_args()
    {"train": train, "evaluate": evaluate, "prepare": prepare}[args.phase](args.arm)
