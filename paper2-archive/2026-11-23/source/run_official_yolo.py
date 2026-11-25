"""Official Ultralytics YOLO horizontal comparison runner on SVRDD7 (640, 100E, seed=42).

Covers 6 official COCO80-pretrained YOLO architectures under unified SVRDD7 Val COCO & Native evaluation:
  - v8n  : YOLOv8n   (yolov8n.pt,  Detect O2M,           AdamW lr0=0.000909 matching upstream Ultralytics auto for nc=7)
  - v9t  : YOLOv9t   (yolov9t.pt,  Detect O2M,           AdamW lr0=0.000909)
  - v10n : YOLOv10n  (yolov10n.pt, v10Detect E2E O2O+O2M,AdamW lr0=0.000909; primary O2O NMS-free + standalone O2M)
  - y11n : YOLO11n   (yolo11n.pt,  Detect O2M,           AdamW lr0=0.000909)
  - y12n : YOLO12n   (yolo12n.pt,  Detect O2M,           AdamW lr0=0.000909)
  - y26s : YOLO26s   (yolo26s.pt,  Detect E2E O2M+O2O,   MuSGD lr0=0.01 matching YOLO26n b0; primary O2M + standalone O2O)
"""
import argparse
import collections
import contextlib
import copy
import csv
import hashlib
import io
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import torch
from ultralytics import YOLO
from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.nn.modules.head import Detect, v10Detect
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import ops
from ultralytics.utils.nms import non_max_suppression
from ultralytics.utils.torch_utils import get_flops, init_seeds, unwrap_model

STAGE = Path(__file__).resolve().parent
sys.path[:0] = [str(STAGE)]
import run_v2
import run_v6
import run_v7
from coco_selection import COCOSelectionTrainer, COCOValidator

sha, dump, now, proof = run_v2.sha, run_v2.dump, run_v2.now, run_v2.proof
reference = run_v2.reference

PRETRAINED_DIR = Path("/root/pretrained/official_yolo_v840")
WEIGHT_FILES = dict(
    v8n="yolov8n.pt",
    v9t="yolov9t.pt",
    v10n="yolov10n.pt",
    y11n="yolo11n.pt",
    y12n="yolo12n.pt",
    y26s="yolo26s.pt",
)
LABELS = dict(
    v8n="YOLOv8n",
    v9t="YOLOv9t",
    v10n="YOLOv10n",
    y11n="YOLO11n",
    y12n="YOLO12n",
    y26s="YOLO26s",
)
ARMS = tuple(WEIGHT_FILES.keys())
PRIMARY_MODE = dict(
    v8n="o2m",
    v9t="o2m",
    v10n="o2o",
    y11n="o2m",
    y12n="o2m",
    y26s="o2m",
)
DUAL_HEAD_ARMS = {"v10n", "y26s"}
OPTIM_SPECS = {
    arm: (
        dict(optimizer="MuSGD", lr0=0.01, lrf=0.01, momentum=0.9, weight_decay=0.0005, warmup_bias_lr=0.0)
        if arm == "y26s"
        else dict(optimizer="AdamW", lr0=0.000909, lrf=0.01, momentum=0.9, weight_decay=0.0005, warmup_bias_lr=0.0)
    )
    for arm in ARMS
}


def safe_as_o2m(model, fuse: bool = False):
    """Return an O2M evaluation/deployment copy for any Detect or v10Detect model."""
    out = copy.deepcopy(model).float().eval()
    head = out.model[-1]
    assert hasattr(head, "cv2") and head.cv2 is not None and head.cv3 is not None, type(head)
    for attr in ("one2one_cv2", "one2one_cv3"):
        if hasattr(head, attr):
            delattr(head, attr)
    if isinstance(head, v10Detect):
        head.__class__ = Detect
    assert not getattr(head, "end2end", False), f"Expected O2M head, got end2end=True on {type(head)}"
    if fuse:
        out.fuse(verbose=False)
    return out


def compact_o2o(model):
    """Return a fused O2O end2end deployment copy (drops O2M cv2/cv3 branches)."""
    out = copy.deepcopy(model).float().eval()
    head = out.model[-1]
    assert getattr(head, "end2end", False), f"Expected end2end=True model, got {type(head)}"
    out.fuse(verbose=False)
    head = out.model[-1]
    head.cv2 = None
    head.cv3 = None
    return out


class OfficialO2MCOCOValidator(COCOValidator):
    """COCOValidator using safe_as_o2m so single-head Detect models do not fail on del one2one_cv2."""

    def __call__(self, trainer=None, model=None):
        if trainer is None:
            m = reference.load_model(model) if isinstance(model, (str, Path)) else model
            model = safe_as_o2m(m, fuse=False)
        return self._call_coco(trainer, model)

    def _call_coco(self, trainer, model):
        old_amp = trainer.amp if trainer is not None else None
        if trainer is not None:
            trainer.amp = False
        try:
            result = DetectionValidator.__call__(self, trainer=trainer, model=model)
        finally:
            if trainer is not None:
                trainer.amp = old_amp
        result.update({f"COCO/{k}": v for k, v in self.coco_main.items()})
        result["fitness"] = self.coco_main["AP50_95"]
        return result


class OfficialO2MTaskValidator(reference.TaskValidator):
    """Standalone O2M validator safe for both single-head Detect and dual-head Detect/v10Detect."""

    def __call__(self, trainer=None, model=None):
        if trainer is None:
            m = reference.load_model(model) if isinstance(model, (str, Path)) else model
            model = safe_as_o2m(m, fuse=False)
        return DetectionValidator.__call__(self, trainer=trainer, model=model)


class OfficialO2OCOCOValidator(run_v7.O2OTaskValidator):
    """COCO-selecting O2O validator for end2end models (e.g. YOLOv10n)."""

    def __call__(self, trainer=None, model=None):
        old_amp = trainer.amp if trainer is not None else None
        if trainer is not None:
            trainer.amp = False
        try:
            result = super().__call__(trainer=trainer, model=model)
        finally:
            if trainer is not None:
                trainer.amp = old_amp
        result.update({f"COCO/{k}": v for k, v in self.coco_main.items()})
        result["fitness"] = self.coco_main["AP50_95"]
        return result

    def init_metrics(self, model):
        super().init_metrics(model)
        assert self.end2end is True, f"Expected end2end=True in O2O validator, got {self.end2end}"
        if not hasattr(self, "ground_truth"):
            self.ground_truth = COCO(str(reference.DATA / "annotations/instances_val.json"))
            self.image_ids = {Path(v["file_name"]).stem: k for k, v in self.ground_truth.imgs.items()}
            self.categories = {v["name"]: k for k, v in self.ground_truth.cats.items()}
            assert len(self.image_ids) == 1000 and set(self.categories) == set(reference.NAMES)
        self.evaluated_ids = set()
        self.coco_main = None

    def update_metrics(self, preds, batch):
        super().update_metrics(preds, batch)
        self.evaluated_ids.update(self.image_ids[Path(p).stem] for p in batch["im_file"])

    def get_stats(self):
        result = super().get_stats()
        if not self.jdict:
            self.coco_main = dict(AP50_95=0.0, AP50=0.0, AP75=0.0, AR100=0.0)
            return result
        predictions = []
        for p in self.jdict:
            index = int(p["category_id"]) - 1
            if 0 <= index < 7 and Path(p["file_name"]).stem in self.image_ids:
                predictions.append(
                    dict(
                        image_id=self.image_ids[Path(p["file_name"]).stem],
                        category_id=self.categories[reference.NAMES[index]],
                        bbox=p["bbox"],
                        score=p["score"],
                    )
                )
        with contextlib.redirect_stdout(io.StringIO()):
            if predictions:
                detections = self.ground_truth.loadRes(predictions)
            else:
                detections = COCO()
                detections.dataset = {**self.ground_truth.dataset, "annotations": []}
                detections.createIndex()
            ev = COCOeval(self.ground_truth, detections, "bbox")
            ev.params.imgIds = sorted(self.evaluated_ids)
            ev.params.catIds = [self.categories[n] for n in reference.NAMES]
            ev.params.maxDets = [1, 10, 100]
            ev.evaluate()
            ev.accumulate()
        self.coco_main = dict(
            AP50_95=reference.ap(ev),
            AP50=reference.ap(ev, iou=0.5),
            AP75=reference.ap(ev, iou=0.75),
            AR100=reference.ar100(ev),
        )
        assert all(v is not None and 0 <= v <= 1 for v in self.coco_main.values())
        return result


def build_candidate(arm: str):
    if arm not in WEIGHT_FILES:
        raise ValueError(f"Unknown arm: {arm}")
    pt_path = PRETRAINED_DIR / WEIGHT_FILES[arm]
    assert pt_path.is_file(), f"Missing pretrained checkpoint: {pt_path}"
    Detect.legacy = False
    init_seeds(42, deterministic=True)
    public = YOLO(str(pt_path)).model.float()
    cand = DetectionModel(copy.deepcopy(public.yaml), nc=7, verbose=False).float()
    cand.load(public, verbose=False)
    cand.names = dict(enumerate(reference.NAMES))

    pub_sd = public.state_dict()
    cand_sd = cand.state_dict()
    eq_keys = [k for k in cand_sd if k in pub_sd and cand_sd[k].shape == pub_sd[k].shape and torch.equal(cand_sd[k], pub_sd[k])]
    diff_keys = [k for k in cand_sd if k not in pub_sd or cand_sd[k].shape != pub_sd[k].shape]

    # Verify 100% of backbone + neck (layers 0..-2) parameters are inherited verbatim from COCO80.
    head_prefix = f"model.{len(cand.model) - 1}."
    bb_neck_diff = [k for k in diff_keys if not k.startswith(head_prefix)]
    assert not bb_neck_diff, f"{arm} failed 100% backbone+neck inheritance: {bb_neck_diff}"

    total_float = sum(cand_sd[k].numel() for k in cand_sd if cand_sd[k].is_floating_point() and not k.endswith("num_batches_tracked"))
    inh_float = sum(cand_sd[k].numel() for k in eq_keys if cand_sd[k].is_floating_point() and not k.endswith("num_batches_tracked"))
    bb_total = sum(
        cand_sd[k].numel()
        for k in cand_sd
        if not k.startswith(head_prefix) and cand_sd[k].is_floating_point() and not k.endswith("num_batches_tracked")
    )
    bb_inh = sum(
        cand_sd[k].numel()
        for k in eq_keys
        if not k.startswith(head_prefix) and cand_sd[k].is_floating_point() and not k.endswith("num_batches_tracked")
    )
    assert bb_inh == bb_total, (bb_inh, bb_total)

    deploy_model = compact_o2o(cand) if PRIMARY_MODE[arm] == "o2o" else safe_as_o2m(cand, fuse=True)
    checks = dict(
        label=LABELS[arm],
        primary_mode=PRIMARY_MODE[arm],
        head_type=type(cand.model[-1]).__name__,
        reg_max=int(cand.model[-1].reg_max),
        end2end=bool(getattr(cand.model[-1], "end2end", False)),
        inherited_tensors=len(eq_keys),
        total_tensors=len(cand_sd),
        reinitialized_tensors=len(diff_keys),
        inherited_numel_ratio=round(inh_float / total_float, 6),
        backbone_neck_inherited_ratio=round(bb_inh / bb_total, 6),
        train_params=sum(p.numel() for p in cand.parameters()),
        deployed_params=sum(p.numel() for p in deploy_model.parameters()),
    )
    return cand, pt_path, checks


def prepare(arm: str):
    cand, pt_path, checks = build_candidate(arm)
    folder = proof(arm)
    folder.mkdir(exist_ok=True, parents=True)

    init = folder / "initial.pt"
    if init.exists():
        assert reference.state_sha(reference.load_model(init)) == reference.state_sha(cand)
    else:
        torch.save(dict(model=cand, ema=None, epoch=-1, optimizer=None), init)
    dump(folder / "candidate_checks.json", checks)

    base_proto = dict(json.loads((run_v2.REF / "protocol.json").read_text())["common_protocol"])
    config = dict(base_proto)
    config.update(OPTIM_SPECS[arm])
    config.update(
        seed=42,
        imgsz=640,
        epochs=100,
        batch=32,
        nbs=64,
        model=str(init),
        data=str(run_v2.DATA_YAML),
        project=str(STAGE / "runs"),
        name=arm,
        exist_ok=False,
    )
    assert (config["batch"], config["epochs"], config["imgsz"], config["seed"]) == (32, 100, 640, 42)
    assert config["mosaic"] == 1.0 and int(config["save_period"]) == 5 and int(config["close_mosaic"]) == 10

    splits = {}
    for sname, expected in (("train", 6000), ("val", 1000)):
        arr = (run_v2.REF / "data" / f"{sname}.txt").read_text().splitlines()
        assert len(arr) == len(set(arr)) == expected and all("/test/" not in x.lower() for x in arr)
        splits[sname] = set(Path(x).name for x in arr)
    assert splits["train"].isdisjoint(splits["val"])

    runtime_files = [
        Path(__file__),
        Path(run_v2.__file__),
        Path(run_v7.__file__),
        pt_path,
        run_v2.DATA_YAML,
        run_v2.REF / "data/train.txt",
        run_v2.REF / "data/val.txt",
        reference.DATA / "annotations/instances_train.json",
        reference.DATA / "annotations/instances_val.json",
        Path(sys.modules["coco_selection"].__file__),
    ]
    contract = dict(
        time=now(),
        arm=arm,
        label=LABELS[arm],
        seed=42,
        imgsz=640,
        primary_mode=PRIMARY_MODE[arm],
        public_pt=str(pt_path),
        public_pt_sha256=sha(pt_path),
        initial=str(init),
        initial_sha256=sha(init),
        state_sha=reference.state_sha(cand),
        params=checks["train_params"],
        deployed_params=checks["deployed_params"],
        optim=OPTIM_SPECS[arm],
        checks=checks,
        quad_best_saving=True,
        hashes={str(p): sha(p) for p in runtime_files},
        protocol=config,
    )
    dump(folder / "contract.json", contract)
    print(
        f"[prepare OK] {arm} ({LABELS[arm]}): head={checks['head_type']}, mode={PRIMARY_MODE[arm]}, "
        f"opt={OPTIM_SPECS[arm]['optimizer']}(lr0={OPTIM_SPECS[arm]['lr0']}), "
        f"params={checks['train_params']:,} -> deploy={checks['deployed_params']:,}, "
        f"inh={checks['inherited_numel_ratio']*100:.2f}% (bb+neck=100%)"
    )
    return contract


def train(arm: str):
    if (STAGE / "runs" / arm).exists():
        raise FileExistsError(f"Existing run: {arm}")
    c = prepare(arm)
    folder = proof(arm)
    expected = collections.Counter(Path(x).name for x in (run_v2.REF / "data/train.txt").read_text().splitlines())
    opt_name = c["optim"]["optimizer"]
    val_cls = OfficialO2OCOCOValidator if PRIMARY_MODE[arm] == "o2o" else OfficialO2MCOCOValidator

    class OfficialQuadBestTrainer(COCOSelectionTrainer):
        def get_model(self, cfg=None, weights=None, verbose=True):
            return reference.load_model(c["initial"])

        def get_validator(self):
            self.loss_names = ("box_loss", "cls_loss", "dfl_loss")
            args = copy.copy(self.args)
            args.save_json = True
            return val_cls(self.test_loader, save_dir=self.save_dir, args=args, _callbacks=self.callbacks)

        def preprocess_batch(self, batch):
            self.exposure.update(Path(p).name for p in batch["im_file"])
            result = super().preprocess_batch(batch)
            if not hasattr(self, "first_receipt"):
                self.first_receipt = run_v2.first_batch(result)
                ref_fb_path = proof("HistoricalCLS") / "first_batch.json"
                matches = False
                if ref_fb_path.is_file():
                    ref_fb = json.loads(ref_fb_path.read_text())
                    matches = all(self.first_receipt[k] == ref_fb[k] for k in ("img", "cls", "bboxes", "batch_idx", "names", "shape"))
                dump(folder / "first_batch.json", dict(**self.first_receipt, matches_historical_cls_seed42=matches))
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
            k50 = (self.validator.coco_main["AP50"], self.validator.coco_main["AP50_95"])
            b50 = getattr(self, "_best_ap50_key", None)
            self._new_ap50_best = b50 is None or k50 > b50
            if self._new_ap50_best:
                self._best_ap50_key = k50

            kn95 = (metrics["metrics/mAP50-95(B)"], metrics["metrics/mAP50(B)"])
            bn95 = getattr(self, "_best_nat95_key", None)
            self._new_nat95_best = bn95 is None or kn95 > bn95
            if self._new_nat95_best:
                self._best_nat95_key = kn95

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
        assert type(t.optimizer).__name__ == opt_name and t.amp and t.batch_size == 32 and t.epochs == 100
        assert t.args.nbs == 64 and t.accumulate == 2 and t.args.seed == 42 and int(t.args.imgsz) == 640
        assert int(t.args.close_mosaic) == 10 and t.args.mosaic == 1.0 and int(t.args.save_period) == 5
        opt_ids = {id(p) for g in t.optimizer.param_groups for p in g["params"]}
        all_ids = {id(p) for p in m.parameters()}
        trainable_ids = {id(p) for p in m.parameters() if p.requires_grad}
        assert trainable_ids.issubset(opt_ids), f"Optimizer param mismatch: missing={len(trainable_ids - opt_ids)}"
        assert opt_ids.issubset(all_ids), f"Unexpected external optimizer params: {len(opt_ids - all_ids)}"
        t.success = t.skips = 0
        m.criterion = m.init_criterion()
        torch.cuda.reset_peak_memory_stats()
        t.peak_alloc = t.peak_res = 0
        dump(
            folder / "runtime.json",
            dict(
                time=now(),
                pid=os.getpid(),
                gpu=torch.cuda.get_device_name(0),
                arm=arm,
                label=LABELS[arm],
                state_sha=c["state_sha"],
                public_pt_sha256=c["public_pt_sha256"],
                seed=42,
                imgsz=640,
                primary_mode=PRIMARY_MODE[arm],
                params=c["params"],
                deployed_params=c["deployed_params"],
                optimizer=type(t.optimizer).__name__,
                lr0=float(t.args.lr0),
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
        assert t.exposure == expected
        crit = unwrap_model(t.model).criterion
        if hasattr(crit, "updates"):
            assert crit.updates == t.epoch + 1
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

    t = OfficialQuadBestTrainer(overrides=c["protocol"])
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
            arm=arm,
            label=LABELS[arm],
            epochs=100,
            seed=42,
            imgsz=640,
            primary_mode=PRIMARY_MODE[arm],
            primary=sum(x["primary"] for x in rows),
            state_sha=c["state_sha"],
            quad_best_saving=True,
        ),
    )


def evaluate_o2m_official(ck: Path, output: Path, imgsz: int = 640, batch: int = 32, data_yaml: Path = run_v2.DATA_YAML):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=False)
    model = reference.load_model(ck)
    parameters = sum(p.numel() for p in model.parameters())
    validator = OfficialO2MTaskValidator(
        args=dict(
            task="detect",
            data=str(data_yaml),
            imgsz=imgsz,
            batch=batch,
            workers=4,
            device="0",
            conf=0.001,
            iou=0.7,
            max_det=300,
            rect=True,
            save_json=True,
            plots=False,
            verbose=False,
            half=False,
            split="val",
        ),
        save_dir=output,
    )
    native = validator(model=model)
    gt = COCO(str(reference.DATA / "annotations/instances_val.json"))
    image_ids = {Path(v["file_name"]).stem: k for k, v in gt.imgs.items()}
    categories = {v["name"]: k for k, v in gt.cats.items()}
    result = run_v7._collect_coco_and_native(
        validator,
        gt,
        image_ids,
        categories,
        output,
        ck,
        parameters,
        imgsz,
        batch,
        eval_desc=f"O2M + class-aware NMS ({imgsz}); COCO maxDets100; Val only",
        warp_cfg=None,
    )
    result["native_metrics"] = native
    dump(output / "metrics.json", result)
    return result


def benchmark_arm_efficiency(arm: str, ck: Path, val_imgs: list[np.ndarray], mode: str) -> dict:
    raw_m = reference.load_model(ck)
    train_params = sum(p.numel() for p in raw_m.parameters())
    dep = compact_o2o(raw_m) if mode == "o2o" else safe_as_o2m(raw_m, fuse=True)
    deploy_params = sum(p.numel() for p in dep.parameters())
    flops_g = float(get_flops(copy.deepcopy(dep).cpu(), imgsz=640))

    dep_fp32 = copy.deepcopy(dep).cuda().eval()
    dep_fp16 = copy.deepcopy(dep).cuda().half().eval()

    def _pure_gpu_ms(model, bs: int, dtype) -> float:
        x = torch.randn(bs, 3, 640, 640, device="cuda", dtype=dtype)
        with torch.no_grad():
            for _ in range(30):
                _ = model(x)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(150):
                _ = model(x)
            torch.cuda.synchronize()
            dt = (time.perf_counter() - t0) * 1000.0 / 150.0
        return dt / bs

    gpu_fp32_bs1 = _pure_gpu_ms(dep_fp32, 1, torch.float32)
    gpu_fp32_bs32 = _pure_gpu_ms(dep_fp32, 32, torch.float32)
    gpu_fp16_bs1 = _pure_gpu_ms(dep_fp16, 1, torch.float16)
    gpu_fp16_bs32 = _pure_gpu_ms(dep_fp16, 32, torch.float16)

    t_pre = t_fwd = t_post = 0.0
    with torch.no_grad():
        for im0 in val_imgs[:20]:
            im_p = cv2.resize(im0, (640, 640), interpolation=cv2.INTER_LINEAR)
            x = torch.from_numpy(np.ascontiguousarray(im_p[:, :, ::-1].transpose(2, 0, 1))).unsqueeze(0).cuda().float() / 255.0
            _ = dep_fp32(x)
        torch.cuda.synchronize()

        t_total0 = time.perf_counter()
        for im0 in val_imgs:
            tp0 = time.perf_counter()
            im_p = cv2.resize(im0, (640, 640), interpolation=cv2.INTER_LINEAR)
            t_in = np.ascontiguousarray(im_p[:, :, ::-1].transpose(2, 0, 1))
            x = torch.from_numpy(t_in).unsqueeze(0).cuda(non_blocking=False).float() * (1.0 / 255.0)
            torch.cuda.synchronize()
            tp1 = time.perf_counter()
            out = dep_fp32(x)
            torch.cuda.synchronize()
            tp2 = time.perf_counter()
            pred = out[0].clone()
            if mode == "o2o":
                dets = non_max_suppression(pred, 0.25, 0.70, max_det=300, end2end=True)
            else:
                dets = non_max_suppression(pred, 0.25, 0.70, multi_label=True, agnostic=False, nc=7, max_det=300)
            _ = dets[0].cpu()
            tp3 = time.perf_counter()
            t_pre += tp1 - tp0
            t_fwd += tp2 - tp1
            t_post += tp3 - tp2
        e2e_ms = (time.perf_counter() - t_total0) * 1000.0 / len(val_imgs)

    del dep_fp32, dep_fp16
    torch.cuda.empty_cache()

    res = dict(
        arm=arm,
        label=LABELS[arm],
        mode=mode,
        imgsz=640,
        gpu=torch.cuda.get_device_name(0),
        n_e2e_imgs=len(val_imgs),
        train_params=train_params,
        deploy_params=deploy_params,
        gflops=round(flops_g, 3),
        gpu_fp32_bs1_ms=round(gpu_fp32_bs1, 3),
        gpu_fp32_bs32_ms=round(gpu_fp32_bs32, 3),
        gpu_fp16_bs1_ms=round(gpu_fp16_bs1, 3),
        gpu_fp16_bs32_ms=round(gpu_fp16_bs32, 3),
        e2e_bs1_ms=round(e2e_ms, 3),
        e2e_bs1_fps=round(1000.0 / e2e_ms, 2),
        e2e_pre_ms=round(t_pre * 1000.0 / len(val_imgs), 3),
        e2e_fwd_ms=round(t_fwd * 1000.0 / len(val_imgs), 3),
        e2e_post_ms=round(t_post * 1000.0 / len(val_imgs), 3),
    )
    print(
        f"  [Bench {arm} ({mode})]: params={train_params}/{deploy_params}, GFLOPs={res['gflops']}, "
        f"GPU32={res['gpu_fp32_bs32_ms']:.3f}ms, E2E1={res['e2e_bs1_ms']:.3f}ms ({res['e2e_bs1_fps']} FPS)"
    )
    return res


def evaluate(arm: str):
    folder = proof(arm)
    assert (folder / "trained.json").is_file(), f"Run {arm} not finished training!"
    result = {}
    epoch_map_path = STAGE / "runs" / arm / "checkpoint_epoch_map.json"
    if epoch_map_path.is_file():
        result["checkpoint_epoch_map"] = json.loads(epoch_map_path.read_text())

    pmode = PRIMARY_MODE[arm]
    primary_fn = run_v7.evaluate_o2o_at_imgsz if pmode == "o2o" else evaluate_o2m_official

    seen_sha = {}
    pred_paths = {}
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
            r = primary_fn(ck, path, imgsz=640, batch=32)
        if (path / "predictions.coco.json").is_file():
            pred_paths[kind] = path / "predictions.coco.json"
        seen_sha[digest] = kind
        result[kind] = r

    # For dual-head architectures (YOLOv10n & YOLO26s), also independently evaluate the secondary branch.
    if arm in DUAL_HEAD_ARMS:
        sec_mode = "o2m" if pmode == "o2o" else "o2o"
        sec_fn = evaluate_o2m_official if sec_mode == "o2m" else run_v7.evaluate_o2o_at_imgsz
        seen_sec = {}
        for kind in ("best", "best_ap50", "last"):
            ck = STAGE / "runs" / arm / "weights" / f"{kind}.pt"
            if not ck.is_file():
                continue
            digest = sha(ck)
            k_sec = f"{sec_mode}_{kind}"
            p_sec = folder / "evaluation" / k_sec
            if (p_sec / "metrics.json").exists():
                rs = json.loads((p_sec / "metrics.json").read_text())
                assert rs["sha256"] == digest
            elif digest in seen_sec:
                rs = dict(result[seen_sec[digest]], checkpoint=str(ck), same_as=seen_sec[digest])
                p_sec.mkdir(parents=True, exist_ok=True)
                dump(p_sec / "metrics.json", rs)
            else:
                rs = sec_fn(ck, p_sec, imgsz=640, batch=32)
            if (p_sec / "predictions.coco.json").is_file():
                pred_paths[k_sec] = p_sec / "predictions.coco.json"
            seen_sec[digest] = k_sec
            result[k_sec] = rs

    dump(folder / "evaluation/summary.json", result)

    slices = {}
    for k, pp in pred_paths.items():
        slices[k] = run_v6.compute_extended_slices(pp)
    dump(folder / "evaluation/slice_metrics.json", slices)

    # Write native_metrics.json including E80-E90 mean
    csv_path = STAGE / "runs" / arm / "results.csv"
    cmap = result.get("checkpoint_epoch_map", {})
    rows = [{k.strip(): v.strip() for k, v in r.items()} for r in csv.DictReader(csv_path.open("r", encoding="utf-8"))]
    eps = [json.loads(l) for l in (folder / "epochs.jsonl").read_text().splitlines() if l.strip()]
    e80_90 = [x["metrics"] for x in eps if 80 <= x["epoch"] <= 90]
    e80_90_mean = {
        k: sum(x[k] for x in e80_90) / len(e80_90)
        for k in (
            "COCO/AP50",
            "COCO/AP75",
            "COCO/AP50_95",
            "COCO/AR100",
            "metrics/mAP50(B)",
            "metrics/AP75(B)",
            "metrics/mAP50-95(B)",
            "metrics/precision(B)",
            "metrics/recall(B)",
        )
        if e80_90 and k in e80_90[0]
    }
    peaks = {}
    for col in ("COCO/AP50_95", "COCO/AP50", "metrics/mAP50-95(B)", "metrics/mAP50(B)"):
        if col in rows[0]:
            br = max(rows, key=lambda r: float(r[col]))
            peaks[col] = {"epoch": int(br["epoch"]), "metric": float(br[col])}
    dump(
        folder / "evaluation/native_metrics.json",
        dict(
            run=arm,
            label=LABELS[arm],
            primary_mode=pmode,
            dataset_split="SVRDD7 Val",
            seed=42,
            imgsz_train=640,
            checkpoint_epoch_map=cmap,
            e80_e90_mean=e80_90_mean,
            train_log_selection_peaks=peaks,
        ),
    )

    # Benchmark RTX 4090 D efficiency on best.pt
    val_files = (run_v2.REF / "data/val.txt").read_text().splitlines()[:200]
    val_imgs = [cv2.imread(p) for p in val_files]
    ck_best = STAGE / "runs" / arm / "weights/best.pt"
    eff = dict(primary=benchmark_arm_efficiency(arm, ck_best, val_imgs, pmode))
    if arm in DUAL_HEAD_ARMS:
        sec_mode = "o2m" if pmode == "o2o" else "o2o"
        eff["secondary"] = benchmark_arm_efficiency(arm, ck_best, val_imgs, sec_mode)
    dump(folder / "evaluation/efficiency.json", eff)

    print(f"[evaluate OK] {arm} ({LABELS[arm]}):")
    for k in ("best", "best_ap50", "last", "o2m_best", "o2o_best"):
        v = result.get(k)
        if isinstance(v, dict) and "main" in v:
            m = v["main"]
            print(f"  {k:<12}: COCO(50={m['AP50']*100:.3f}, 75={m['AP75']*100:.3f}, 95={m['AP50_95']*100:.3f}, AR={m['AR100']*100:.3f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("prepare", "train", "evaluate"))
    parser.add_argument("arm", choices=ARMS)
    args = parser.parse_args()
    {"prepare": prepare, "train": train, "evaluate": evaluate}[args.phase](args.arm)
