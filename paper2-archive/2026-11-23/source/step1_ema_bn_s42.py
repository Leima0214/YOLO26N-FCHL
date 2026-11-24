"""Step 1-1: Weight Averaging (EMA windows) + Cumulative BN Recalibration (Seed 42).

Compares BEFORE vs AFTER exact cumulative BN statistics recalibration (1 clean FP32 pass
over all 6,000 SVRDD Train images with Mosaic/flip/HSV disabled) on:
  - HistCLS_SchemeA_P2 (windows E60_85, E65_90, E60_90 + single best.pt sanity check)
  - HistoricalCLS_DualBest (windows E60_85, E65_90, E60_90)
  - p2_det (auto-included if finished & trained.json + epoch85/90 exist)

Evaluates every variant via official reference.evaluate_checkpoint (FP32 O2M + class-aware
NMS 0.70, COCO maxDets=100) AND exact custom narrow-box (<8px / <16px) slices.
"""
import argparse
import copy
import datetime as dt
import json
import sys
from pathlib import Path

import torch
import yaml
from pycocotools.coco import COCO

STAGE = Path("/root/YOLO26N-FCHL/reports/paper2_cls_candidate_diagnosis_640_20261120_42106")
ROOT = Path("/root/YOLO26N-FCHL")
GEO = ROOT / "reports/paper2_geometry_semantic_640_20261109_42266"
REF = ROOT / "reports/paper2_cls_p2p3_snake_640_20261115_42264"
sys.path[:0] = [
    str(STAGE),
    str(GEO / "source"),
    str(REF / "source/dependencies"),
    str(GEO / "source/dependencies"),
    str(ROOT),
]

import models  # noqa: F401
import experiment as reference
import run_v2
import candidates_v3  # noqa: F401
import candidates_v5  # noqa: F401
import step1_cpu_health_audit as cpu_audit
from ultralytics.cfg import get_cfg
from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.data.utils import check_det_dataset

WINDOWS = {
    "E60_85": [60, 65, 70, 75, 80, 85],
    "E65_90": [65, 70, 75, 80, 85, 90],
    "E60_90": [60, 65, 70, 75, 80, 85, 90],
}

EXPECTED_PRE_RECAL_E60_85 = {
    "HistCLS_SchemeA_P2": dict(AP50_95=43.709, AP50=70.250),
    "HistoricalCLS_DualBest": dict(AP50_95=43.288, AP50=69.855),
}


def build_clean_train_loader(batch: int = 32):
    tmp_dir = STAGE / "diagnostics/_tmp_ema_bn"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    clean_yaml = tmp_dir / "svrdd7_clean_train.yaml"
    base_cfg = yaml.safe_load(run_v2.DATA_YAML.read_text())
    base_cfg["val"] = str(REF / "data/train.txt")
    clean_yaml.write_text(yaml.safe_dump(base_cfg, sort_keys=False))

    data_dict = check_det_dataset(str(clean_yaml))
    cfg = get_cfg(overrides=dict(task="detect", imgsz=640, batch=batch, rect=False, workers=4, data=str(clean_yaml)))
    ds = build_yolo_dataset(cfg, str(REF / "data/train.txt"), batch=batch, data=data_dict, mode="val", stride=32)
    assert len(ds) == 6000, f"Expected 6000 clean train images, got {len(ds)}"
    loader = build_dataloader(ds, batch=batch, workers=4, shuffle=False, rank=-1, drop_last=False, pin_memory=True)
    return loader


def average_checkpoints(arm: str, epochs: list[int]):
    models_loaded = []
    shas = {}
    for ep in epochs:
        ck = STAGE / "runs" / arm / "weights" / f"epoch{ep}.pt"
        assert ck.is_file(), f"Missing snapshot {ck}"
        shas[str(ep)] = reference.file_sha(ck)[:12]
        models_loaded.append(reference.load_model(ck))

    avg_model = copy.deepcopy(models_loaded[-1]).float()
    sds = [m.state_dict() for m in models_loaded]
    ref_keys = list(sds[-1].keys())
    new_sd = {}
    n_float = 0
    n_nonfloat = 0
    for k in ref_keys:
        t0 = sds[-1][k]
        if t0.is_floating_point():
            acc = torch.zeros_like(t0, dtype=torch.float64)
            for sd in sds:
                acc.add_(sd[k].to(dtype=torch.float64))
            new_sd[k] = (acc / float(len(sds))).to(dtype=t0.dtype)
            n_float += 1
        else:
            new_sd[k] = t0.clone()
            n_nonfloat += 1
    avg_model.load_state_dict(new_sd, strict=True)
    return avg_model, dict(snapshots=epochs, snapshot_shas_12=shas, float_tensors=n_float, non_float_from_last=n_nonfloat)


def recalibrate_bn_exact(model: torch.nn.Module, loader, device: str = "cuda:0") -> tuple[torch.nn.Module, dict]:
    """Recalibrate ONLY BatchNorm2d running_mean/running_var via exact cumulative averaging."""
    recal = copy.deepcopy(model).float().to(device)
    recal.eval()  # Keep DropPath/Dropout/PKSBlock/Detect in eval mode globally!

    orig_momentum = {}
    bn_mods = []
    for name, mod in recal.named_modules():
        if isinstance(mod, torch.nn.BatchNorm2d):
            orig_momentum[name] = mod.momentum
            mod.reset_running_stats()
            mod.momentum = None  # Exact cumulative moving average 1/N
            mod.train()
            bn_mods.append((name, mod))

    assert len(bn_mods) > 0, "Expected BatchNorm2d modules in model!"
    n_imgs = 0
    n_batches = 0
    with torch.no_grad():
        for batch in loader:
            imgs = batch["img"].to(device, non_blocking=True).float() / 255.0
            _ = recal(imgs)
            n_imgs += int(imgs.shape[0])
            n_batches += 1

    assert n_imgs == 6000, f"Expected 6000 images in BN pass, saw {n_imgs}"
    for name, mod in bn_mods:
        assert int(mod.num_batches_tracked.item()) == n_batches, (
            f"BN layer {name} not updated: {mod.num_batches_tracked.item()} != {n_batches}"
        )
        mod.momentum = orig_momentum[name]
    recal.eval()
    recal.cpu()
    head_idx = len(recal.model) - 1
    meta = dict(
        bn_layers_recalibrated=len(bn_mods),
        clean_train_images=n_imgs,
        batches=n_batches,
        p2_fusion_bn_tracked=sum(1 for n, _ in bn_mods if "p2_fusion" in n),
        detect_head_bn_tracked=sum(1 for n, _ in bn_mods if n.startswith(f"model.{head_idx}.")),
    )
    return recal, meta


def save_and_evaluate(model: torch.nn.Module, ckpt_out: Path, eval_dir: Path, gt: COCO, cats_by_name: dict) -> dict:
    ckpt_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(model=copy.deepcopy(model).cpu().float(), ema=None, epoch=-1, optimizer=None), ckpt_out)
    res = reference.evaluate_checkpoint(ckpt_out, eval_dir, run_v2.DATA_YAML)
    preds = json.loads((eval_dir / "predictions.coco.json").read_text())
    dt_coco = cpu_audit.build_coco_dt(gt, preds)
    slices = cpu_audit.evaluate_all_slices(gt, dt_coco, cats_by_name)
    res["slices"] = slices
    return res


def fmt_main_pct(res: dict) -> dict:
    m = res["main"]
    return dict(
        AP50_95=round(m["AP50_95"] * 100.0, 3),
        AP50=round(m["AP50"] * 100.0, 3),
        AP75=round(m["AP75"] * 100.0, 3),
        AP_small=round(m["AP_small"] * 100.0, 3),
        AP_medium=round(m["AP_medium"] * 100.0, 3),
        AP_large=round(m["AP_large"] * 100.0, 3),
        AR100=round(m["AR100"] * 100.0, 3),
        P=round(m["P"] * 100.0, 3),
        Recall=round(m["Recall"] * 100.0, 3),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arms", nargs="*", default=[])
    args = parser.parse_args()

    arms = list(args.arms) if args.arms else ["HistCLS_SchemeA_P2", "HistoricalCLS_DualBest"]
    if not args.arms:
        p2_det_trained = STAGE / "records/p2_det/trained.json"
        p2_det_e90 = STAGE / "runs/p2_det/weights/epoch90.pt"
        if p2_det_trained.is_file() and p2_det_e90.is_file():
            arms.append("p2_det")

    gt = COCO(str(cpu_audit.ANN_VAL))
    cats_by_name = {c["name"]: c["id"] for c in gt.cats.values()}
    loader = build_clean_train_loader(batch=32)

    tmp_root = STAGE / "diagnostics/_tmp_ema_bn/runs"
    out = dict(
        generated_at=dt.datetime.now().isoformat(),
        seed=42,
        protocol="SVRDD7 Val 1,000 imgs / 2,612 GTs, FP32 O2M + class-aware NMS 0.70, BN cumulative 1-pass recalibration on 6,000 clean Train imgs",
        arms={},
    )

    for arm in arms:
        arm_res = dict(windows={}, single_ckpt_sanity={})
        print(f"=== EMA + BN Recalibration: {arm} ===", flush=True)
        for wname, epochs in WINDOWS.items():
            pre_model, avg_meta = average_checkpoints(arm, epochs)
            pre_ck = tmp_root / arm / f"ema_{wname}_pre.pt"
            pre_eval = tmp_root / arm / f"eval_{wname}_pre"
            if pre_eval.exists():
                import shutil
                shutil.rmtree(pre_eval)
            pre_res = save_and_evaluate(pre_model, pre_ck, pre_eval, gt, cats_by_name)
            pre_main = fmt_main_pct(pre_res)

            # Assert exact reproduction vs prior EMA6 measurement
            if wname == "E60_85" and arm in EXPECTED_PRE_RECAL_E60_85:
                exp = EXPECTED_PRE_RECAL_E60_85[arm]
                d95 = abs(pre_main["AP50_95"] - exp["AP50_95"])
                d50 = abs(pre_main["AP50"] - exp["AP50"])
                assert d95 < 0.01 and d50 < 0.01, f"Pre-recal EMA6 parity mismatch on {arm}: {pre_main} vs {exp}"
                print(f"  [EMA6 Pre-Recal Parity OK: {arm}] diff95={d95:.4f}, diff50={d50:.4f}", flush=True)

            recal_model, bn_meta = recalibrate_bn_exact(pre_model, loader)
            post_ck = STAGE / "runs" / arm / "weights" / f"ema_{wname}_bn_recal.pt"
            post_eval = tmp_root / arm / f"eval_{wname}_post"
            if post_eval.exists():
                import shutil
                shutil.rmtree(post_eval)
            post_res = save_and_evaluate(recal_model, post_ck, post_eval, gt, cats_by_name)
            post_main = fmt_main_pct(post_res)
            delta = {k: round(post_main[k] - pre_main[k], 3) for k in ("AP50_95", "AP50", "AP75", "AP_small", "AR100")}

            print(
                f"  [{arm} {wname}] PRE : 95={pre_main['AP50_95']:.3f} 50={pre_main['AP50']:.3f} 75={pre_main['AP75']:.3f} S={pre_main['AP_small']:.3f} lt8={pre_res['slices']['short640_lt8']['AP50_95']:.3f}",
                flush=True,
            )
            print(
                f"  [{arm} {wname}] POST: 95={post_main['AP50_95']:.3f} ({delta['AP50_95']:+.3f}) 50={post_main['AP50']:.3f} ({delta['AP50']:+.3f}) 75={post_main['AP75']:.3f} ({delta['AP75']:+.3f}) lt8={post_res['slices']['short640_lt8']['AP50_95']:.3f} -> {post_ck.name}",
                flush=True,
            )

            arm_res["windows"][wname] = dict(
                averaging=avg_meta,
                bn_recalibration=bn_meta,
                recalibrated_checkpoint=str(post_ck),
                recalibrated_sha256=post_res["sha256"],
                pre_recal=dict(main=pre_main, slices=pre_res["slices"], classes=pre_res["classes"]),
                post_recal=dict(main=post_main, slices=post_res["slices"], classes=post_res["classes"]),
                delta_post_minus_pre=delta,
            )

        # Sanity check on single best.pt checkpoint of HistCLS_SchemeA_P2
        if arm == "HistCLS_SchemeA_P2":
            best_ck = STAGE / "runs" / arm / "weights/best.pt"
            best_m = reference.load_model(best_ck)
            ref_best_metrics = json.loads((STAGE / "records" / arm / "evaluation/best/metrics.json").read_text())
            best_pre_main = fmt_main_pct(ref_best_metrics)
            best_recal, bn_meta_single = recalibrate_bn_exact(best_m, loader)
            s_ck = STAGE / "runs" / arm / "weights/best_bn_recal_sanity.pt"
            s_eval = tmp_root / arm / "eval_best_sanity_post"
            if s_eval.exists():
                import shutil
                shutil.rmtree(s_eval)
            s_res = save_and_evaluate(best_recal, s_ck, s_eval, gt, cats_by_name)
            s_main = fmt_main_pct(s_res)
            s_delta = {k: round(s_main[k] - best_pre_main[k], 3) for k in ("AP50_95", "AP50", "AP75", "AP_small", "AR100")}
            print(
                f"  [Sanity best.pt] PRE: 95={best_pre_main['AP50_95']:.3f} 50={best_pre_main['AP50']:.3f} -> "
                f"POST-BN: 95={s_main['AP50_95']:.3f} ({s_delta['AP50_95']:+.3f}) 50={s_main['AP50']:.3f} ({s_delta['AP50']:+.3f})",
                flush=True,
            )
            arm_res["single_ckpt_sanity"]["best_E90"] = dict(
                bn_recalibration=bn_meta_single,
                pre_recal_official_main=best_pre_main,
                post_recal=dict(main=s_main, slices=s_res["slices"], classes=s_res["classes"]),
                delta_post_minus_pre=s_delta,
            )

        out["arms"][arm] = arm_res

    out_path = STAGE / "diagnostics/step1_ema_bn_s42.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"[DONE] Saved {out_path}", flush=True)


if __name__ == "__main__":
    main()
