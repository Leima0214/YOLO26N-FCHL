"""V6 training & unified 640+768 evaluation strictly from CLS-P2A (October HistoricalCLS + SchemeA_P2):
- b1     : Backbone L0->L2 SPD + DEConv zero-init shallow high-frequency side-branch (640, seed=42)
- warp50 : Single-breakpoint 50% road vertical resampling (256/384, pre-Mosaic 1024->640, exact inverse-y before NMS & COCO eval, seed=42)
- b1_w50 : Factorial combination of b1 + warp50 (seed=42)
- Five checkpoints saved per arm: best.pt (COCO AP50:95), best_ap50.pt (COCO AP50),
  best_native.pt, best_native_ap50.pt, last.pt (E100), plus epochXX.pt snapshots (save_period=5).
"""
import argparse
import collections
import copy
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO
import torch

STAGE = Path(__file__).resolve().parent
sys.path[:0] = [str(STAGE)]
import run_v2
import run_v4
import candidates_v3
import candidates_v5
import candidates_v6
import step1_cpu_health_audit as audit_mod
from coco_selection import COCOSelectionTrainer, COCOValidator
from ultralytics.utils.torch_utils import unwrap_model

sha, dump, now, proof = run_v2.sha, run_v2.dump, run_v2.now, run_v2.proof
reference = run_v2.reference


class InverseWarpCOCOValidator(candidates_v6.InverseWarpValidatorMixin, COCOValidator):
    warp_split_y = candidates_v6.WARP50_SPLIT_Y
    warp_top_frac = candidates_v6.WARP50_TOP_FRAC


class InverseWarpTaskValidator(candidates_v6.InverseWarpValidatorMixin, reference.TaskValidator):
    warp_split_y = candidates_v6.WARP50_SPLIT_Y
    warp_top_frac = candidates_v6.WARP50_TOP_FRAC


def evaluate_warp_at_imgsz(
    ck: Path,
    output: Path,
    imgsz: int = 640,
    batch: int = 32,
    split_y: float = candidates_v6.WARP50_SPLIT_Y,
    top_frac: float = candidates_v6.WARP50_TOP_FRAC,
    data_yaml: Path = run_v2.DATA_YAML,
):
    """Exact mirror of reference.evaluate_checkpoint / run_v4.evaluate_at_imgsz with pre-NMS inverse-y warp mapping."""
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=False)
    model = reference.load_model(ck)
    parameters = sum(p.numel() for p in model.parameters())

    class _ConfiguredWarpValidator(candidates_v6.InverseWarpValidatorMixin, reference.TaskValidator):
        warp_split_y = float(split_y)
        warp_top_frac = float(top_frac)

    validator = _ConfiguredWarpValidator(
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
    classes = []
    present = list(validator.metrics.box.ap_class_index)
    for i, name in enumerate(reference.NAMES):
        e = reference.coco_evaluate(gt, predictions, [categories[name]])
        j = present.index(i) if i in present else None
        classes.append(
            {
                "class": name,
                "AP50": reference.ap(e, iou=0.5),
                "AP75": reference.ap(e, iou=0.75),
                "AP50_95": reference.ap(e),
                "AR100": reference.ar100(e),
                "P": float(validator.metrics.box.p[j]) if j is not None else None,
                "Recall": float(validator.metrics.box.r[j]) if j is not None else None,
            }
        )
    result = dict(
        time=now(),
        checkpoint=str(ck),
        sha256=sha(ck),
        imgsz=imgsz,
        batch=batch,
        warp=dict(split_y=split_y, top_frac=top_frac),
        training_parameters=parameters,
        predictions=len(predictions),
        evaluation="O2M + class-aware NMS (inverse-mapped to original 1024x1024 pre-NMS); COCO maxDets100; Val only",
        main=main,
        classes=classes,
        native_metrics=native,
    )
    dump(output / "metrics.json", result)
    return result


def compute_extended_slices(pred_json_path: Path) -> dict:
    """Compute exact original-annotation COCO slices including orientation & vertical road zones."""
    gt = COCO(str(reference.DATA / "annotations/instances_val.json"))
    cats_by_name = {v["name"]: k for k, v in gt.cats.items()}
    all_cats = [cats_by_name[n] for n in reference.NAMES]
    preds = json.loads(pred_json_path.read_text())
    dt_coco = audit_mod.build_coco_dt(gt, preds)

    out = audit_mod.evaluate_all_slices(gt, dt_coco, cats_by_name)
    extra_specs = [
        ("horiz_flat_h_lt_w", lambda w, h, a: h < w, all_cats),
        ("horiz_flat_lt16", lambda w, h, a: h < w and min(w, h) * audit_mod.GAIN_640 < 16.0, all_cats),
        ("vert_narrow_w_lt_h", lambda w, h, a: w < h, all_cats),
        ("vert_narrow_lt16", lambda w, h, a: w < h and min(w, h) * audit_mod.GAIN_640 < 16.0, all_cats),
        ("lower_road_yc_ge512", lambda w, h, a: (a["bbox"][1] + 0.5 * h) >= 512.0, all_cats),
        ("upper_half_yc_lt512", lambda w, h, a: (a["bbox"][1] + 0.5 * h) < 512.0, all_cats),
        ("road_band_512_768", lambda w, h, a: 512.0 <= (a["bbox"][1] + 0.5 * h) < 768.0, all_cats),
        ("lower_road_lt16", lambda w, h, a: (a["bbox"][1] + 0.5 * h) >= 512.0 and min(w, h) * audit_mod.GAIN_640 < 16.0, all_cats),
    ]
    for sname, pred_fn, cids in extra_specs:
        m, n_gt, n_imgs = audit_mod.eval_custom_slice(gt, dt_coco, pred_fn, pred_fn, cids)
        out[sname] = {
            "n_gt": n_gt,
            "n_gt_images": n_imgs,
            **{k: (round(v * 100.0, 3) if v is not None else None) for k, v in m.items()},
        }
    return out


def prepare(arm: str):
    if arm not in candidates_v6.ARMS:
        raise ValueError(f"Unknown v6 arm: {arm}")
    base = run_v2.load_historical_base()
    folder = proof(arm)
    folder.mkdir(exist_ok=True, parents=True)
    model, checks = candidates_v6.prepare(base, arm)

    # Assert bit-for-bit parent identity vs HistCLS_SchemeA_P2/initial.pt on all CLS-P2A weights
    p2a_init = proof("HistCLS_SchemeA_P2") / "initial.pt"
    assert p2a_init.is_file(), f"Missing {p2a_init}"
    p2a_sd = reference.load_model(p2a_init).state_dict()
    cur_sd = model.state_dict()
    assert set(p2a_sd.keys()).issubset(set(cur_sd.keys())), "Missing CLS-P2A keys!"
    for k, v in p2a_sd.items():
        assert torch.equal(v, cur_sd[k]), f"HistCLS_SchemeA_P2 tensor mismatch on {k}!"
    checks["checks"]["inherits_schemeA_p2_initial_exact"] = True
    if not candidates_v6.has_b1(arm):
        assert set(p2a_sd.keys()) == set(cur_sd.keys())
    if arm == "b1_w50" and (proof("b1") / "initial.pt").is_file():
        b1_init = reference.load_model(proof("b1") / "initial.pt")
        assert reference.state_sha(model) == reference.state_sha(b1_init), "b1_w50 initial state_sha differs from b1!"
        checks["checks"]["matches_b1_initial_exact"] = True

    init = folder / "initial.pt"
    if init.exists():
        assert reference.state_sha(reference.load_model(init)) == reference.state_sha(model)
    else:
        torch.save(dict(model=model, ema=None, epoch=-1, optimizer=None), init)
    dump(folder / "candidate_checks.json", checks)

    seed = candidates_v6.SEEDS[arm]
    overrides = dict(candidates_v6.PROTOCOL_OVERRIDES[arm])
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
    assert (config["batch"], config["epochs"], config["imgsz"], config["seed"]) == (32, 100, 640, seed)
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
        Path(candidates_v6.__file__),
        Path(candidates_v5.__file__),
        Path(candidates_v3.__file__),
        Path(run_v2.__file__),
        Path(run_v4.__file__),
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
        seed=seed,
        parent_family="CLS-P2A (HistoricalCLS + SchemeA_P2)",
        parent_initial=str(p2a_init),
        parent_initial_sha256=sha(p2a_init),
        parent_hist_initial_sha256=run_v2.HIST_INIT_SHA,
        parent_state_digests=run_v2.EXPECTED_HIST,
        initial=str(init),
        initial_sha256=sha(init),
        state_sha=reference.state_sha(model),
        params=sum(p.numel() for p in model.parameters()),
        added_over_p2a=checks["added_over_p2a"],
        added_params=checks["added"],
        protocol_overrides=overrides,
        warp=dict(split_y=candidates_v6.WARP50_SPLIT_Y, top_frac=candidates_v6.WARP50_TOP_FRAC) if candidates_v6.has_warp50(arm) else None,
        quad_best_saving=True,
        auto_eval_resolutions=[640, 768],
        hashes={str(p): sha(p) for p in runtime_files},
        protocol=config,
    )
    dump(folder / "contract.json", contract)
    print(
        f"[prepare OK] {arm} (seed={seed}, parent=CLS-P2A): "
        f"params={contract['params']} (+{checks['added_over_p2a']} over CLS-P2A / +{checks['added']} over HistCLS), "
        f"deployed_o2m={checks['checks']['deployed_o2m_params']}"
    )
    return contract


def train(arm: str):
    if (STAGE / "runs" / arm).exists():
        raise FileExistsError(f"Existing run: {arm}")
    c = prepare(arm)
    folder = proof(arm)
    seed = c["seed"]
    expected = collections.Counter(Path(x).name for x in (run_v2.REF / "data/train.txt").read_text().splitlines())

    class QuadBestV6Trainer(COCOSelectionTrainer):
        def get_model(self, cfg=None, weights=None, verbose=True):
            return reference.load_model(c["initial"])

        def build_dataset(self, img_path: str, mode: str = "train", batch: int | None = None):
            if not candidates_v6.has_warp50(arm):
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
                warp_split_y=candidates_v6.WARP50_SPLIT_Y,
                warp_top_frac=candidates_v6.WARP50_TOP_FRAC,
            )

        def get_validator(self):
            if not candidates_v6.has_warp50(arm):
                return super().get_validator()
            self.loss_names = ("box_loss", "cls_loss", "dfl_loss")
            args = copy.copy(self.args)
            args.save_json = True
            return InverseWarpCOCOValidator(
                self.test_loader, save_dir=self.save_dir, args=args, _callbacks=self.callbacks
            )

        def preprocess_batch(self, batch):
            self.exposure.update(Path(p).name for p in batch["im_file"])
            result = super().preprocess_batch(batch)
            if not hasattr(self, "first_receipt"):
                self.first_receipt = run_v2.first_batch(result)
                ref_fb = json.loads((proof("HistoricalCLS") / "first_batch.json").read_text())
                if not candidates_v6.has_warp50(arm):
                    for k in ("img", "cls", "bboxes", "batch_idx", "names", "shape"):
                        assert self.first_receipt[k] == ref_fb[k], f"First batch mismatch vs HistoricalCLS seed42 on {k}!"
                    dump(folder / "first_batch.json", dict(**self.first_receipt, matches_historical_cls_seed42=True))
                else:
                    for k in ("names", "shape"):
                        assert self.first_receipt[k] == ref_fb[k], f"First batch warp50 mismatch on {k}!"
                    dump(
                        folder / "first_batch.json",
                        dict(
                            **self.first_receipt,
                            matches_historical_cls_seed42_names_shape=True,
                            warp_split_y=candidates_v6.WARP50_SPLIT_Y,
                            warp_top_frac=candidates_v6.WARP50_TOP_FRAC,
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
        if candidates_v6.has_warp50(arm):
            assert isinstance(t.train_loader.dataset, candidates_v6.Warp50YOLODataset)
            assert isinstance(t.test_loader.dataset, candidates_v6.Warp50YOLODataset)
        assert type(t.optimizer).__name__ == "MuSGD" and t.amp and t.batch_size == 32 and t.epochs == 100
        assert t.args.nbs == 64 and t.accumulate == 2 and t.args.seed == seed
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
                parent_family="CLS-P2A",
                parent_initial_sha256=c["parent_initial_sha256"],
                state_sha=c["state_sha"],
                seed=seed,
                close_mosaic=10,
                save_period=int(t.args.save_period),
                params=c["params"],
                added_over_p2a=c["added_over_p2a"],
                added_params=c["added_params"],
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

    t = QuadBestV6Trainer(overrides=c["protocol"])
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
            close_mosaic=10,
            primary=sum(x["primary"] for x in rows),
            parent_family="CLS-P2A",
            state_sha=c["state_sha"],
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

    eval_fn_640 = (
        (lambda ck, path: evaluate_warp_at_imgsz(ck, path, imgsz=640, batch=32))
        if candidates_v6.has_warp50(arm)
        else (lambda ck, path: reference.evaluate_checkpoint(ck, path, run_v2.DATA_YAML))
    )
    eval_fn_768 = (
        (lambda ck, path: evaluate_warp_at_imgsz(ck, path, imgsz=768, batch=32))
        if candidates_v6.has_warp50(arm)
        else (lambda ck, path: run_v4.evaluate_at_imgsz(ck, path, imgsz=768, batch=32))
    )

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
            r = eval_fn_640(ck, path)
        if (path / "predictions.coco.json").is_file():
            pred_paths[kind] = path / "predictions.coco.json"
        seen_sha[digest] = kind
        result[kind] = r

    seen_768 = {}
    for kind in ("best", "best_ap50"):
        ck = STAGE / "runs" / arm / "weights" / f"{kind}.pt"
        if not ck.is_file():
            continue
        digest = sha(ck)
        key768 = f"{kind}_768"
        path768 = folder / "evaluation" / key768
        if (path768 / "metrics.json").exists():
            r768 = json.loads((path768 / "metrics.json").read_text())
            assert r768["sha256"] == digest
        elif digest in seen_768:
            r768 = dict(result[seen_768[digest]], checkpoint=str(ck), same_as=seen_768[digest])
            path768.mkdir(parents=True, exist_ok=True)
            dump(path768 / "metrics.json", r768)
        else:
            r768 = eval_fn_768(ck, path768)
        if (path768 / "predictions.coco.json").is_file():
            pred_paths[key768] = path768 / "predictions.coco.json"
        seen_768[digest] = key768
        result[key768] = r768

    if candidates_v6.has_warp50(arm):
        ck_best = STAGE / "runs" / arm / "weights/best.pt"
        if ck_best.is_file():
            digest_best = sha(ck_best)
            path_u768 = folder / "evaluation/best_768_uniform"
            if (path_u768 / "metrics.json").exists():
                ru768 = json.loads((path_u768 / "metrics.json").read_text())
                assert ru768["sha256"] == digest_best
            else:
                if path_u768.exists():
                    path_u768 = path_u768.with_name("best_768_uniform_" + str(time.time_ns()))
                ru768 = run_v4.evaluate_at_imgsz(ck_best, path_u768, imgsz=768, batch=32)
            if (path_u768 / "predictions.coco.json").is_file():
                pred_paths["best_768_uniform"] = path_u768 / "predictions.coco.json"
            result["best_768_uniform"] = ru768

    dump(folder / "evaluation/summary.json", result)

    # Original-annotation custom COCO slices for every distinct evaluated prediction file
    slices = {}
    for k, pp in pred_paths.items():
        slices[k] = compute_extended_slices(pp)
    dump(folder / "evaluation/slice_metrics.json", slices)

    print(f"[evaluate OK] {arm}:")
    for k in ("best", "best_ap50", "best_native", "best_native_ap50", "last", "best_768", "best_ap50_768", "best_768_uniform"):
        if k in result:
            m = result[k]["main"]
            base_k = k.replace("_768_uniform", "").replace("_768", "")
            ep = result.get("checkpoint_epoch_map", {}).get(base_k, 100 if base_k == "last" else "?")
            print(
                f"  {k:<17} (E{ep}): AP50:95={m['AP50_95']*100:.3f}%, AP50={m['AP50']*100:.3f}%, "
                f"AP75={m['AP75']*100:.3f}%, S={m['AP_small']*100:.3f}%, M={m['AP_medium']*100:.3f}%, "
                f"L={m['AP_large']*100:.3f}%, AR100={m['AR100']*100:.3f}%"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("prepare", "train", "evaluate"))
    parser.add_argument("arm", choices=candidates_v6.ARMS)
    args = parser.parse_args()
    {"prepare": prepare, "train": train, "evaluate": evaluate}[args.phase](args.arm)
