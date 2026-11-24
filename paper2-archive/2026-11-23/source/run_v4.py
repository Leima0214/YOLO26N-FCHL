"""V4 training & unified 640+768 evaluation strictly on original October HistoricalCLS:
- QuadBest saving: best.pt (COCO AP50:95), best_ap50.pt (COCO AP50), best_native.pt, best_native_ap50.pt, last.pt (E100)
- Seed-paired first-batch verification:
    * HistoricalCLS_seed100      -> records seed100 first_batch (verifies distinct from seed42)
    * HistCLS_SchemeA_P2_seed100 -> asserts exact match vs HistoricalCLS_seed100 first_batch
    * HistCLS_SchemeA_P2P3       -> asserts exact match vs HistoricalCLS (seed42) first_batch
- Unified official FP32 COCO evaluation at BOTH 640 (all 5 checkpoints) AND 768 (batch=32 on best & distinct best_ap50).
"""
import argparse
import collections
import json
import os
import sys
import time
from pathlib import Path
import torch
from pycocotools.coco import COCO

STAGE = Path(__file__).resolve().parent
sys.path[:0] = [str(STAGE)]
import run_v2
import candidates_v3
import candidates_v4
from coco_selection import COCOSelectionTrainer
from ultralytics.utils.torch_utils import unwrap_model

sha, dump, now, proof = run_v2.sha, run_v2.dump, run_v2.now, run_v2.proof
reference = run_v2.reference


def prepare(arm: str):
    if arm not in candidates_v4.ARMS:
        raise ValueError(f"Unknown v4 arm: {arm}")
    base = run_v2.load_historical_base()
    folder = proof(arm)
    folder.mkdir(exist_ok=True, parents=True)
    model, checks = candidates_v4.prepare(base, arm)

    # Extra check for SchemeA_P2P3: verify P2 context matches HistCLS_SchemeA_P2 (seed42) bit-for-bit!
    if arm == "HistCLS_SchemeA_P2P3" and (proof("HistCLS_SchemeA_P2") / "initial.pt").is_file():
        p2_ref = reference.load_model(proof("HistCLS_SchemeA_P2") / "initial.pt").state_dict()
        p2p3_sd = model.state_dict()
        for k, v in p2_ref.items():
            assert k in p2p3_sd and torch.equal(v, p2p3_sd[k]), f"SchemeA_P2 tensor mismatch on {k}!"
        checks["checks"]["inherits_schemeA_p2_seed42_exact"] = True

    init = folder / "initial.pt"
    if init.exists():
        assert reference.state_sha(reference.load_model(init)) == reference.state_sha(model)
    else:
        torch.save(dict(model=model, ema=None, epoch=-1, optimizer=None), init)
    dump(folder / "candidate_checks.json", checks)

    seed = candidates_v4.SEEDS[arm]
    config = dict(json.loads((run_v2.REF / "protocol.json").read_text())["common_protocol"])
    config.update(
        seed=seed,
        model=str(init),
        data=str(run_v2.DATA_YAML),
        project=str(STAGE / "runs"),
        name=arm,
        exist_ok=False,
    )
    assert (config["batch"], config["epochs"], config["imgsz"], config["seed"]) == (32, 100, 640, seed)
    assert config["mosaic"] == 1.0 and config["close_mosaic"] == 10

    splits = {}
    for name, expected in (("train", 6000), ("val", 1000)):
        arr = (run_v2.REF / "data" / f"{name}.txt").read_text().splitlines()
        assert len(arr) == len(set(arr)) == expected and all("/test/" not in x.lower() for x in arr)
        splits[name] = set(Path(x).name for x in arr)
    assert splits["train"].isdisjoint(splits["val"])

    runtime_files = [
        Path(__file__),
        Path(candidates_v4.__file__),
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
        seed=seed,
        parent_family="HistoricalCLS (October byte-exact original)",
        parent_initial=str(run_v2.HIST_INIT),
        parent_initial_sha256=run_v2.HIST_INIT_SHA,
        parent_state_digests=run_v2.EXPECTED_HIST,
        initial=str(init),
        initial_sha256=sha(init),
        state_sha=reference.state_sha(model),
        params=sum(p.numel() for p in model.parameters()),
        added_params=checks["added"],
        quad_best_saving=True,
        auto_eval_resolutions=[640, 768],
        hashes={str(p): sha(p) for p in runtime_files},
        protocol=config,
    )
    dump(folder / "contract.json", contract)
    print(
        f"[prepare OK] {arm} (seed={seed}, parent=HistoricalCLS {run_v2.OLD_ALL[:10]}): "
        f"params={contract['params']} (+{checks['added']}), deployed_o2m={checks['checks']['deployed_o2m_params']}"
    )
    return contract


def train(arm: str):
    if (STAGE / "runs" / arm).exists():
        raise FileExistsError(f"Existing run: {arm}")
    c = prepare(arm)
    folder = proof(arm)
    seed = c["seed"]
    expected = collections.Counter(Path(x).name for x in (run_v2.REF / "data/train.txt").read_text().splitlines())

    class QuadBestTrainer(COCOSelectionTrainer):
        def get_model(self, cfg=None, weights=None, verbose=True):
            return reference.load_model(c["initial"])

        def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
            if mode != "train" or int(self.args.seed) == 42:
                return super().get_dataloader(dataset_path, batch_size=batch_size, rank=rank, mode=mode)
            from ultralytics.data.build import InfiniteDataLoader, seed_worker
            from ultralytics.utils.torch_utils import torch_distributed_zero_first

            with torch_distributed_zero_first(rank):
                dataset = self.build_dataset(dataset_path, mode, batch_size)
            batch = min(batch_size, len(dataset))
            nd = torch.cuda.device_count()
            nw = min(os.cpu_count() // max(nd, 1), self.args.workers)
            generator = torch.Generator()
            # Ultralytics build_dataloader hardcodes 6148914691236517205 + RANK (ignoring args.seed for shuffle & worker RNG!).
            # Derive a deterministic seed-dependent generator so seed=100 truly changes batch order AND mosaic/aug RNG,
            # while pairing identically between HistoricalCLS_seed100 and HistCLS_SchemeA_P2_seed100.
            dl_seed = (6148914691236517205 + rank + (int(self.args.seed) - 42) * 1000003) & ((1 << 63) - 1)
            generator.manual_seed(dl_seed)
            self._v4_dataloader_seed = int(dl_seed)
            return InfiniteDataLoader(
                dataset=dataset,
                batch_size=batch,
                shuffle=True,
                num_workers=nw,
                sampler=None,
                prefetch_factor=4 if nw > 0 else None,
                pin_memory=nd > 0,
                collate_fn=getattr(dataset, "collate_fn", None),
                worker_init_fn=seed_worker,
                generator=generator,
                drop_last=self.args.compile,
            )

        def preprocess_batch(self, batch):
            self.exposure.update(Path(p).name for p in batch["im_file"])
            result = super().preprocess_batch(batch)
            if not hasattr(self, "first_receipt"):
                self.first_receipt = run_v2.first_batch(result)
                meta = {"dataloader_generator_seed": getattr(self, "_v4_dataloader_seed", 6148914691236517205)}
                if arm == "HistoricalCLS_seed100":
                    s42 = json.loads((proof("HistoricalCLS") / "first_batch.json").read_text())
                    assert self.first_receipt["names"] != s42["names"], "seed=100 produced identical order to seed=42!"
                    assert self.first_receipt["img"] != s42["img"], "seed=100 produced identical pixels to seed=42!"
                    meta["distinct_from_seed42"] = True
                elif arm == "HistCLS_SchemeA_P2_seed100":
                    ref_fb = json.loads((proof("HistoricalCLS_seed100") / "first_batch.json").read_text())
                    for k in ("img", "cls", "bboxes", "batch_idx", "names", "shape"):
                        assert self.first_receipt[k] == ref_fb[k], f"First batch mismatch vs HistoricalCLS_seed100 on {k}!"
                    meta["matches_paired_cls_seed100"] = True
                elif arm == "HistCLS_SchemeA_P2P3":
                    ref_fb = json.loads((proof("HistoricalCLS") / "first_batch.json").read_text())
                    for k in ("img", "cls", "bboxes", "batch_idx", "names", "shape"):
                        assert self.first_receipt[k] == ref_fb[k], f"First batch mismatch vs HistoricalCLS seed42 on {k}!"
                    meta["matches_historical_cls_seed42"] = True
                dump(folder / "first_batch.json", dict(**self.first_receipt, **meta))
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
        assert type(t.optimizer).__name__ == "MuSGD" and t.amp and t.batch_size == 32 and t.epochs == 100
        assert t.args.nbs == 64 and t.accumulate == 2 and t.args.seed == seed
        assert t.args.close_mosaic == 10 and t.args.mosaic == 1.0
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
                seed=seed,
                params=c["params"],
                added_params=c["added_params"],
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
            seed=seed,
            primary=sum(x["primary"] for x in rows),
            parent_family="HistoricalCLS",
            state_sha=c["state_sha"],
            quad_best_saving=True,
        ),
    )


def evaluate_at_imgsz(ck: Path, output: Path, imgsz: int = 768, batch: int = 32):
    """Exact mirror of reference.evaluate_checkpoint with configurable imgsz (official batch=32, rect, FP32 O2M)."""
    output.mkdir(parents=True, exist_ok=True)
    model = reference.load_model(ck)
    parameters = sum(p.numel() for p in model.parameters())
    validator = reference.TaskValidator(
        args=dict(
            task="detect",
            data=str(run_v2.DATA_YAML),
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
        P=float(native.get("metrics/precision(B)", 0.0)),
        Recall=float(native.get("metrics/recall(B)", 0.0)),
    )
    classes = []
    for i, name in enumerate(reference.NAMES):
        c_all = pr[:, :, i, 0, 2]
        c_50 = pr[0, :, i, 0, 2]
        c_75 = pr[5, :, i, 0, 2]
        c_l = pr[:, :, i, 3, 2]
        c_r = rc[:, i, 0, 2]
        classes.append(
            {
                "class": name,
                "AP50": float(c_50[c_50 > -1].mean()),
                "AP75": float(c_75[c_75 > -1].mean()),
                "AP50_95": float(c_all[c_all > -1].mean()),
                "AP_large": float(c_l[c_l > -1].mean()) if (c_l > -1).any() else None,
                "AR100": float(c_r[c_r > -1].mean()),
            }
        )
    report = dict(
        time=now(),
        checkpoint=str(ck),
        sha256=sha(ck),
        imgsz=imgsz,
        batch=batch,
        parameters=parameters,
        predictions=len(predictions),
        main=main,
        classes=classes,
    )
    dump(output / "metrics.json", report)
    return report


def evaluate(arm: str):
    folder = proof(arm)
    assert (folder / "trained.json").is_file()
    result = {}
    epoch_map_path = STAGE / "runs" / arm / "checkpoint_epoch_map.json"
    if epoch_map_path.is_file():
        result["checkpoint_epoch_map"] = json.loads(epoch_map_path.read_text())

    # 1. Official 640 evaluation on all 5 saved weights (SHA-deduplicated)
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

    # 2. Official 768 (batch=32) evaluation on best.pt (and best_ap50.pt if distinct SHA)
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
            r768 = evaluate_at_imgsz(ck, path768, imgsz=768, batch=32)
        seen_768[digest] = key768
        result[key768] = r768

    dump(folder / "evaluation/summary.json", result)
    print(f"[evaluate OK] {arm}:")
    for k in ("best", "best_ap50", "best_native", "best_native_ap50", "last", "best_768", "best_ap50_768"):
        if k in result:
            m = result[k]["main"]
            base_k = k.replace("_768", "")
            ep = result.get("checkpoint_epoch_map", {}).get(base_k, 100 if base_k == "last" else "?")
            print(
                f"  {k:<16} (E{ep}): AP50:95={m['AP50_95']*100:.3f}%, AP50={m['AP50']*100:.3f}%, "
                f"AP75={m['AP75']*100:.3f}%, S={m['AP_small']*100:.3f}%, M={m['AP_medium']*100:.3f}%, "
                f"L={m['AP_large']*100:.3f}%, AR100={m['AR100']*100:.3f}%"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("prepare", "train", "evaluate"))
    parser.add_argument("arm", choices=candidates_v4.ARMS)
    args = parser.parse_args()
    {"prepare": prepare, "train": train, "evaluate": evaluate}[args.phase](args.arm)
