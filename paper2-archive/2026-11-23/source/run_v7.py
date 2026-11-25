"""V7 training & unified dual-metric (COCO + Native) evaluation runner.

Supports:
  - Uniform 640 / 768 arms: b0 (O2M + O2O), b0_p2, b1_768
  - Piecewise-linear vertical warp arms: b0_w50 (256/384, O2M + O2O), b1_w448 (192/448)
  - Quad-best checkpoint saving: best.pt (COCO AP50:95), best_ap50.pt (COCO AP50),
    best_native.pt, best_native_ap50.pt, last.pt (E100), plus save_period=5 snapshots.
"""
import argparse
import collections
import contextlib
import copy
import csv
import io
import json
import os
import sys
import time
from pathlib import Path

from pycocotools.coco import COCO
import torch

STAGE = Path(__file__).resolve().parent
sys.path[:0] = [str(STAGE)]
import run_v2
import run_v6
import candidates_v5
import candidates_v6
import candidates_v7
from coco_selection import COCOSelectionTrainer, COCOValidator
from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.utils import ops
from ultralytics.utils.torch_utils import unwrap_model

sha, dump, now, proof = run_v2.sha, run_v2.dump, run_v2.now, run_v2.proof
reference = run_v2.reference


def load_b0_ref():
    return reference.load_model(candidates_v7.B0_INIT)


def load_public_yolo26n():
    return reference.load_model(candidates_v7.PUBLIC_YOLO26N)


class O2OTaskValidator(DetectionValidator):
    """Native end-to-end (NMS-free) YOLO26 validator returning exact metrics including AP75."""

    def get_stats(self):
        result = super().get_stats()
        values = self.metrics.box.all_ap
        result["metrics/AP75(B)"] = float(values[:, 5].mean()) if len(values) else 0.0
        return result


def make_validators(split_y: float, top_frac: float):
    """Build parameterized InverseWarp validators bound to a specific (split_y, top_frac) geometry."""

    class _WarpCOCOValidator(candidates_v6.InverseWarpValidatorMixin, COCOValidator):
        warp_split_y = float(split_y)
        warp_top_frac = float(top_frac)

    class _WarpO2MTaskValidator(candidates_v6.InverseWarpValidatorMixin, reference.TaskValidator):
        warp_split_y = float(split_y)
        warp_top_frac = float(top_frac)

    class _WarpO2OTaskValidator(candidates_v6.InverseWarpValidatorMixin, O2OTaskValidator):
        warp_split_y = float(split_y)
        warp_top_frac = float(top_frac)

        def postprocess(self, preds):
            # O2O preds[0] is already (bs, max_det, 6) in warped xyxy + conf + cls
            raw_pred = preds[0].clone() if isinstance(preds, (list, tuple)) else preds.clone()
            batch = getattr(self, "_cur_warp_batch", None)
            if batch is not None:
                for si in range(raw_pred.shape[0]):
                    valid_mask = raw_pred[si, :, 4] > 0
                    if valid_mask.any():
                        raw_pred[si, valid_mask, :4] = self._inverse_xyxy_to_orig(
                            raw_pred[si, valid_mask, :4], batch["ori_shape"][si], batch["ratio_pad"][si]
                        )
            return DetectionValidator.postprocess(self, raw_pred)

    return _WarpCOCOValidator, _WarpO2MTaskValidator, _WarpO2OTaskValidator


def _collect_coco_and_native(validator, gt, image_ids, categories, output: Path, ck: Path, parameters: int, imgsz: int, batch: int, eval_desc: str, warp_cfg=None):
    predictions = []
    for p in validator.jdict:
        index = int(p["category_id"]) - 1
        assert 0 <= index < 7 and Path(p["file_name"]).stem in image_ids
        predictions.append(
            dict(
                image_id=image_ids[Path(p["file_name"]).stem],
                category_id=categories[reference.NAMES[index]],
                bbox=p["bbox"],
                score=p["score"],
            )
        )
    dump(output / "predictions.coco.json", predictions)
    if predictions:
        ev = reference.coco_evaluate(gt, predictions, [categories[n] for n in reference.NAMES])
        pr = ev.eval["precision"]
        rc = ev.eval["recall"]
        main = dict(
            AP50=reference.ap(ev, iou=0.5),
            AP75=reference.ap(ev, iou=0.75),
            AP50_95=reference.ap(ev),
            AP_small=float(pr[:, :, :, 1, 2][pr[:, :, :, 1, 2] > -1].mean()),
            AP_medium=float(pr[:, :, :, 2, 2][pr[:, :, :, 2, 2] > -1].mean()),
            AP_large=float(pr[:, :, :, 3, 2][pr[:, :, :, 3, 2] > -1].mean()),
            AR100=float(rc[:, :, 0, 2][rc[:, :, 0, 2] > -1].mean()),
            P=float(validator.metrics.box.mp),
            Recall=float(validator.metrics.box.mr),
        )
    else:
        main = dict(AP50=0.0, AP75=0.0, AP50_95=0.0, AP_small=0.0, AP_medium=0.0, AP_large=0.0, AR100=0.0, P=0.0, Recall=0.0)

    classes = []
    present = list(validator.metrics.box.ap_class_index)
    for i, name in enumerate(reference.NAMES):
        j = present.index(i) if i in present else None
        if predictions:
            e = reference.coco_evaluate(gt, predictions, [categories[name]])
            c_dict = {
                "class": name,
                "AP50": reference.ap(e, iou=0.5),
                "AP75": reference.ap(e, iou=0.75),
                "AP50_95": reference.ap(e),
                "AR100": reference.ar100(e),
                "P": float(validator.metrics.box.p[j]) if j is not None else None,
                "Recall": float(validator.metrics.box.r[j]) if j is not None else None,
            }
        else:
            c_dict = {"class": name, "AP50": 0.0, "AP75": 0.0, "AP50_95": 0.0, "AR100": 0.0, "P": None, "Recall": None}
        classes.append(c_dict)

    return dict(
        time=now(),
        checkpoint=str(ck),
        sha256=sha(ck),
        imgsz=imgsz,
        batch=batch,
        warp=(dict(split_y=warp_cfg[0], top_frac=warp_cfg[1]) if warp_cfg is not None else None),
        training_parameters=parameters,
        predictions=len(predictions),
        evaluation=eval_desc,
        main=main,
        classes=classes,
    )


def evaluate_uniform_at_imgsz(
    ck: Path,
    output: Path,
    imgsz: int = 640,
    batch: int = 32,
    data_yaml: Path = run_v2.DATA_YAML,
):
    """Exact mirror of reference.evaluate_checkpoint at any imgsz, preserving native_metrics + COCO."""
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=False)
    model = reference.load_model(ck)
    parameters = sum(p.numel() for p in model.parameters())
    validator = reference.TaskValidator(
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
    assert set(categories) == set(reference.NAMES)
    result = _collect_coco_and_native(
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


def evaluate_o2o_at_imgsz(
    ck: Path,
    output: Path,
    imgsz: int = 640,
    batch: int = 32,
    warp_cfg: tuple[float, float] | None = None,
    data_yaml: Path = run_v2.DATA_YAML,
):
    """Evaluate native end-to-end (O2O NMS-free) branch for B0 / B0+#5 at original 1024 coords."""
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=False)
    model = reference.load_model(ck)
    assert getattr(model.model[-1], "end2end", False), f"Expected end2end=True model, got {type(model.model[-1])}"
    parameters = sum(p.numel() for p in model.parameters())
    args = dict(
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
    )
    if warp_cfg is None:
        validator = O2OTaskValidator(args=args, save_dir=output)
        desc = f"O2O NMS-free end2end ({imgsz}); COCO maxDets100; Val only"
    else:
        _, _, WarpO2OV = make_validators(warp_cfg[0], warp_cfg[1])
        validator = WarpO2OV(args=args, save_dir=output)
        desc = f"O2O NMS-free end2end (inverse-mapped to 1024x1024, {imgsz}); COCO maxDets100; Val only"

    native = validator(model=model)
    assert validator.end2end is True, "O2O validator unexpectedly disabled end2end!"
    gt = COCO(str(reference.DATA / "annotations/instances_val.json"))
    image_ids = {Path(v["file_name"]).stem: k for k, v in gt.imgs.items()}
    categories = {v["name"]: k for k, v in gt.cats.items()}
    assert set(categories) == set(reference.NAMES)
    result = _collect_coco_and_native(
        validator,
        gt,
        image_ids,
        categories,
        output,
        ck,
        parameters,
        imgsz,
        batch,
        eval_desc=desc,
        warp_cfg=warp_cfg,
    )
    result["native_metrics"] = native
    dump(output / "metrics.json", result)
    return result


def evaluate_warp_at_imgsz(
    ck: Path,
    output: Path,
    imgsz: int = 640,
    batch: int = 32,
    split_y: float = 0.5,
    top_frac: float = 0.4,
    data_yaml: Path = run_v2.DATA_YAML,
):
    return run_v6.evaluate_warp_at_imgsz(
        ck,
        output,
        imgsz=imgsz,
        batch=batch,
        split_y=split_y,
        top_frac=top_frac,
        data_yaml=data_yaml,
    )


def prepare(arm: str):
    if arm not in candidates_v7.ARMS:
        raise ValueError(f"Unknown v7 arm: {arm}")
    hist_base = run_v2.load_historical_base()
    b0_ref = load_b0_ref()
    pub_n = load_public_yolo26n()
    folder = proof(arm)
    folder.mkdir(exist_ok=True, parents=True)

    model, checks = candidates_v7.prepare(
        hist_base,
        b0_ref,
        pub_n,
        arm,
        file_sha_fn=sha,
        state_sha_fn=reference.state_sha,
    )

    if candidates_v7.is_b1_family(arm) and (proof("b1") / "initial.pt").is_file():
        b1_init = reference.load_model(proof("b1") / "initial.pt")
        assert reference.state_sha(model) == reference.state_sha(b1_init), f"{arm} state_sha differs from b1!"
        checks["checks"]["matches_b1_initial_exact"] = True
    if candidates_v7.is_b0_family(arm):
        assert reference.state_sha(model) == reference.state_sha(b0_ref)

    init = folder / "initial.pt"
    if init.exists():
        assert reference.state_sha(reference.load_model(init)) == reference.state_sha(model)
    else:
        torch.save(dict(model=model, ema=None, epoch=-1, optimizer=None), init)
    dump(folder / "candidate_checks.json", checks)

    seed = candidates_v7.SEEDS[arm]
    imgsz = candidates_v7.IMGSZ[arm]
    overrides = dict(candidates_v7.PROTOCOL_OVERRIDES[arm])
    config = dict(json.loads((run_v2.REF / "protocol.json").read_text())["common_protocol"])
    config.update(overrides)
    config.update(
        seed=seed,
        model=str(init),
        data=str(run_v2.DATA_YAML),
        project=str(STAGE / "runs"),
        name=arm,
        exist_ok=False,
    )
    assert (config["batch"], config["epochs"], config["imgsz"], config["seed"]) == (32, 100, imgsz, seed)
    assert config["mosaic"] == 1.0 and int(config["save_period"]) == 5
    assert int(config["close_mosaic"]) == 10 and float(config["flipud"]) == 0.0

    splits = {}
    for name, expected in (("train", 6000), ("val", 1000)):
        arr = (run_v2.REF / "data" / f"{name}.txt").read_text().splitlines()
        assert len(arr) == len(set(arr)) == expected and all("/test/" not in x.lower() for x in arr)
        splits[name] = set(Path(x).name for x in arr)
    assert splits["train"].isdisjoint(splits["val"])

    runtime_files = [
        Path(__file__),
        Path(candidates_v7.__file__),
        Path(candidates_v6.__file__),
        Path(candidates_v5.__file__),
        Path(run_v2.__file__),
        Path(run_v6.__file__),
        STAGE / "historical_head.py",
        candidates_v7.B0_INIT,
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
    wcfg = candidates_v7.warp_cfg(arm)
    contract = dict(
        time=now(),
        arm=arm,
        seed=seed,
        imgsz=imgsz,
        parent_desc=checks["parent_desc"],
        b0_initial_sha256=sha(candidates_v7.B0_INIT),
        public_yolo26n_sha256=sha(candidates_v7.PUBLIC_YOLO26N),
        parent_hist_initial_sha256=run_v2.HIST_INIT_SHA,
        initial=str(init),
        initial_sha256=sha(init),
        state_sha=reference.state_sha(model),
        params=sum(p.numel() for p in model.parameters()),
        deployed_o2m_params=checks["checks"]["deployed_o2m_params"],
        protocol_overrides=overrides,
        warp=(dict(split_y=wcfg[0], top_frac=wcfg[1]) if wcfg is not None else None),
        quad_best_saving=True,
        auto_eval_resolutions=([768, 640] if imgsz == 768 else [640, 768]),
        hashes={str(p): sha(p) for p in runtime_files},
        protocol=config,
    )
    dump(folder / "contract.json", contract)
    print(
        f"[prepare OK] {arm} (seed={seed}, imgsz={imgsz}, warp={wcfg}): "
        f"params={contract['params']}, deployed_o2m={contract['deployed_o2m_params']}"
    )
    return contract


def train(arm: str):
    if (STAGE / "runs" / arm).exists():
        raise FileExistsError(f"Existing run: {arm}")
    c = prepare(arm)
    folder = proof(arm)
    seed = c["seed"]
    imgsz = c["imgsz"]
    wcfg = candidates_v7.warp_cfg(arm)
    expected = collections.Counter(Path(x).name for x in (run_v2.REF / "data/train.txt").read_text().splitlines())

    WarpCOCOV = make_validators(wcfg[0], wcfg[1])[0] if wcfg is not None else None

    class QuadBestV7Trainer(COCOSelectionTrainer):
        def get_model(self, cfg=None, weights=None, verbose=True):
            return reference.load_model(c["initial"])

        def build_dataset(self, img_path: str, mode: str = "train", batch: int | None = None):
            if wcfg is None:
                return super().build_dataset(img_path, mode=mode, batch=batch)
            gs = max(int(unwrap_model(self.model).stride.max() if self.model else 0), 32)
            from ultralytics.utils import colorstr

            return candidates_v6.Warp50YOLODataset(
                img_path=img_path,
                imgsz=self.args.imgsz,
                batch_size=batch,
                augment=mode == "train",
                hyp=self.args,
                rect=self.args.rect or (mode == "val"),
                cache=self.args.cache or None,
                single_cls=self.args.single_cls or False,
                stride=gs,
                pad=0.0 if mode == "train" else 0.5,
                prefix=colorstr(f"{mode}: "),
                task=self.args.task,
                classes=self.args.classes,
                data=self.data,
                fraction=self.args.fraction if mode == "train" else 1.0,
                warp_split_y=wcfg[0],
                warp_top_frac=wcfg[1],
            )

        def get_validator(self):
            if WarpCOCOV is None:
                return super().get_validator()
            self.loss_names = ("box_loss", "cls_loss", "dfl_loss")
            args = copy.copy(self.args)
            args.save_json = True
            return WarpCOCOV(
                self.test_loader, save_dir=self.save_dir, args=args, _callbacks=self.callbacks
            )

        def preprocess_batch(self, batch):
            self.exposure.update(Path(p).name for p in batch["im_file"])
            result = super().preprocess_batch(batch)
            if not hasattr(self, "first_receipt"):
                self.first_receipt = run_v2.first_batch(result)
                ref_fb = json.loads((proof("HistoricalCLS") / "first_batch.json").read_text())
                if wcfg is None and imgsz == 640:
                    for k in ("img", "cls", "bboxes", "batch_idx", "names", "shape"):
                        assert self.first_receipt[k] == ref_fb[k], f"First batch mismatch vs HistoricalCLS seed42 on {k}!"
                    dump(folder / "first_batch.json", dict(**self.first_receipt, matches_historical_cls_seed42=True))
                elif wcfg is None and imgsz == 768:
                    assert self.first_receipt["names"] == ref_fb["names"]
                    assert self.first_receipt["shape"] == [32, 3, 768, 768]
                    dump(folder / "first_batch.json", dict(**self.first_receipt, matches_historical_cls_seed42_names=True))
                else:
                    for k in ("names", "shape"):
                        assert self.first_receipt[k] == ref_fb[k], f"First batch warp mismatch on {k}!"
                    dump(
                        folder / "first_batch.json",
                        dict(
                            **self.first_receipt,
                            matches_historical_cls_seed42_names_shape=True,
                            warp_split_y=wcfg[0],
                            warp_top_frac=wcfg[1],
                        ),
                    )
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
        if wcfg is not None:
            assert isinstance(t.train_loader.dataset, candidates_v6.Warp50YOLODataset)
            assert isinstance(t.test_loader.dataset, candidates_v6.Warp50YOLODataset)
        assert type(t.optimizer).__name__ == "MuSGD" and t.amp and t.batch_size == 32 and t.epochs == 100
        assert t.args.nbs == 64 and t.accumulate == 2 and t.args.seed == seed and int(t.args.imgsz) == imgsz
        assert int(t.args.close_mosaic) == 10 and t.args.mosaic == 1.0 and int(t.args.save_period) == 5
        assert float(t.args.flipud) == 0.0
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
                parent_desc=c["parent_desc"],
                state_sha=c["state_sha"],
                seed=seed,
                imgsz=imgsz,
                close_mosaic=10,
                save_period=int(t.args.save_period),
                params=c["params"],
                deployed_o2m_params=c["deployed_o2m_params"],
                warp=c["warp"],
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

    t = QuadBestV7Trainer(overrides=c["protocol"])
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
            seed=seed,
            imgsz=imgsz,
            close_mosaic=10,
            primary=sum(x["primary"] for x in rows),
            parent_desc=c["parent_desc"],
            state_sha=c["state_sha"],
            quad_best_saving=True,
        ),
    )


def _write_native_metrics(arm: str, summary: dict):
    folder = proof(arm)
    sum_path = folder / "evaluation/summary.json"
    csv_path = STAGE / "runs" / arm / "results.csv"
    cmap = summary.get("checkpoint_epoch_map", {})
    rows = [{k.strip(): v.strip() for k, v in r.items()} for r in csv.DictReader(csv_path.open("r", encoding="utf-8"))]
    eps = [json.loads(l) for l in (folder / "epochs.jsonl").read_text().splitlines() if l.strip()]

    ind = {}
    for k, v in summary.items():
        if k == "checkpoint_epoch_map" or not isinstance(v, dict) or "native_metrics" not in v:
            continue
        base_k = k.replace("o2o_", "").replace("_768_uniform", "").replace("_768", "").replace("_640", "")
        ep = cmap.get(base_k, 100 if base_k == "last" else None)
        ind[k] = {
            "epoch": ep,
            "imgsz": v.get("imgsz"),
            "warp": v.get("warp"),
            "checkpoint_sha256": v.get("sha256"),
            "metrics": v["native_metrics"],
        }

    sel_eps = sorted({ep for ep in cmap.values() if isinstance(ep, int)} | {100})
    train_log = {}
    for ep_num in sel_eps:
        r = [x for x in rows if int(x["epoch"]) == ep_num][0]
        train_log[f"E{ep_num}"] = {
            "metrics/precision(B)": float(r["metrics/precision(B)"]),
            "metrics/recall(B)": float(r["metrics/recall(B)"]),
            "metrics/mAP50(B)": float(r["metrics/mAP50(B)"]),
            "metrics/AP75(B)": float(r["metrics/AP75(B)"]),
            "metrics/mAP50-95(B)": float(r["metrics/mAP50-95(B)"]),
        }

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
        if k in e80_90[0]
    }
    peaks = {}
    for col in ("COCO/AP50_95", "COCO/AP50", "metrics/mAP50-95(B)", "metrics/mAP50(B)"):
        br = max(rows, key=lambda r: float(r[col]))
        peaks[col] = {"epoch": int(br["epoch"]), "metric": float(br[col])}

    doc = {
        "run": arm,
        "dataset_split": "SVRDD7 Val",
        "seed": candidates_v7.SEEDS[arm],
        "imgsz_train": candidates_v7.IMGSZ[arm],
        "metric_units": "fractions on [0,1]",
        "source_sha256": {
            "evaluation/summary.json": sha(sum_path),
            "results.csv": sha(csv_path),
        },
        "checkpoint_epoch_map": cmap,
        "independent_reevaluation_native_metrics": ind,
        "train_log_native_metrics": train_log,
        "e80_e90_mean": e80_90_mean,
        "train_log_selection_peaks": peaks,
    }
    dump(folder / "evaluation/native_metrics.json", doc)


def evaluate(arm: str):
    folder = proof(arm)
    assert (folder / "trained.json").is_file()
    result = {}
    epoch_map_path = STAGE / "runs" / arm / "checkpoint_epoch_map.json"
    if epoch_map_path.is_file():
        result["checkpoint_epoch_map"] = json.loads(epoch_map_path.read_text())

    wcfg = candidates_v7.warp_cfg(arm)
    primary_sz = candidates_v7.IMGSZ[arm]
    diag_sz = 640 if primary_sz == 768 else 768

    def _make_o2m_fn(sz: int):
        if wcfg is not None:
            return lambda ck, path: evaluate_warp_at_imgsz(ck, path, imgsz=sz, batch=32, split_y=wcfg[0], top_frac=wcfg[1])
        return lambda ck, path: evaluate_uniform_at_imgsz(ck, path, imgsz=sz, batch=32)

    eval_primary = _make_o2m_fn(primary_sz)
    eval_diag = _make_o2m_fn(diag_sz)

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
            if path.exists():
                path = path.with_name(kind + "_" + str(time.time_ns()))
            r = eval_primary(ck, path)
        if (path / "predictions.coco.json").is_file():
            pred_paths[kind] = path / "predictions.coco.json"
        seen_sha[digest] = kind
        result[kind] = r

    seen_diag = {}
    for kind in ("best", "best_ap50"):
        ck = STAGE / "runs" / arm / "weights" / f"{kind}.pt"
        if not ck.is_file():
            continue
        digest = sha(ck)
        k_diag = f"{kind}_{diag_sz}"
        p_diag = folder / "evaluation" / k_diag
        if (p_diag / "metrics.json").exists():
            r_d = json.loads((p_diag / "metrics.json").read_text())
            assert r_d["sha256"] == digest
        elif digest in seen_diag:
            r_d = dict(result[seen_diag[digest]], checkpoint=str(ck), same_as=seen_diag[digest])
            p_diag.mkdir(parents=True, exist_ok=True)
            dump(p_diag / "metrics.json", r_d)
        else:
            r_d = eval_diag(ck, p_diag)
        if (p_diag / "predictions.coco.json").is_file():
            pred_paths[k_diag] = p_diag / "predictions.coco.json"
        seen_diag[digest] = k_diag
        result[k_diag] = r_d

    if wcfg is not None:
        ck_best = STAGE / "runs" / arm / "weights/best.pt"
        if ck_best.is_file():
            digest_best = sha(ck_best)
            path_u768 = folder / "evaluation/best_768_uniform"
            if (path_u768 / "metrics.json").exists():
                ru768 = json.loads((path_u768 / "metrics.json").read_text())
                assert ru768["sha256"] == digest_best
            else:
                ru768 = evaluate_uniform_at_imgsz(ck_best, path_u768, imgsz=768, batch=32)
            if (path_u768 / "predictions.coco.json").is_file():
                pred_paths["best_768_uniform"] = path_u768 / "predictions.coco.json"
            result["best_768_uniform"] = ru768

    if candidates_v7.is_b0_family(arm):
        seen_o2o = {}
        for kind in ("best", "best_ap50", "best_native", "last"):
            ck = STAGE / "runs" / arm / "weights" / f"{kind}.pt"
            if not ck.is_file():
                continue
            digest = sha(ck)
            k_o2o = f"o2o_{kind}"
            p_o2o = folder / "evaluation" / k_o2o
            if (p_o2o / "metrics.json").exists():
                ro = json.loads((p_o2o / "metrics.json").read_text())
                assert ro["sha256"] == digest
            elif digest in seen_o2o:
                ro = dict(result[seen_o2o[digest]], checkpoint=str(ck), same_as=seen_o2o[digest])
                p_o2o.mkdir(parents=True, exist_ok=True)
                dump(p_o2o / "metrics.json", ro)
            else:
                ro = evaluate_o2o_at_imgsz(ck, p_o2o, imgsz=primary_sz, batch=32, warp_cfg=wcfg)
            if (p_o2o / "predictions.coco.json").is_file():
                pred_paths[k_o2o] = p_o2o / "predictions.coco.json"
            seen_o2o[digest] = k_o2o
            result[k_o2o] = ro

    dump(folder / "evaluation/summary.json", result)

    slices = {}
    for k, pp in pred_paths.items():
        slices[k] = run_v6.compute_extended_slices(pp)
    dump(folder / "evaluation/slice_metrics.json", slices)
    _write_native_metrics(arm, result)

    print(f"[evaluate OK] {arm}:")
    for k, v in result.items():
        if k == "checkpoint_epoch_map" or not isinstance(v, dict) or "main" not in v:
            continue
        m = v["main"]
        nm = v.get("native_metrics", {})
        print(
            f"  {k:<18}: COCO(50={m['AP50']*100:.3f}, 75={m['AP75']*100:.3f}, 95={m['AP50_95']*100:.3f}, AR={m['AR100']*100:.3f}) "
            f"| NAT(50={nm.get('metrics/mAP50(B)',0)*100:.3f}, 95={nm.get('metrics/mAP50-95(B)',0)*100:.3f})"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("prepare", "train", "evaluate"))
    parser.add_argument("arm", choices=candidates_v7.ARMS)
    args = parser.parse_args()
    defer_flag = STAGE / "runs" / args.arm / "DEFERRED_BY_USER_TOMORROW.txt"
    if args.phase == "train" and defer_flag.is_file():
        print(f"[DEFERRED_BY_USER_TOMORROW] Cleanly halting queue before {args.arm}: {defer_flag}", flush=True)
        sys.exit(99)
    {"prepare": prepare, "train": train, "evaluate": evaluate}[args.phase](args.arm)
