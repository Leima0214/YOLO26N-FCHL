"""Zero-retrain EMA window averaging + 6,000-image clean BN recalibration + score-preserving box voting replay for b1 (seed=42).

Strictly imports candidates_v6 first so ShallowSideC3k2 / ShallowSPDDEConvSide / DEConv2dBNAct unpickle cleanly.
Reuses step1_ema_bn_s42 helpers and step1_box_fusion_replay_s42.replay_checkpoint (<= 0.005 pp native NMS parity gate).
Note: 5-epoch snapshots use Ultralytics 0-indexed filenames epoch{k}.pt = 1-indexed Epoch k+1 (e.g., epoch60..85 = E61..E86).
"""
import datetime as dt
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torchvision
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
import candidates_v6  # noqa: F401
import run_v6
import step1_cpu_health_audit as cpu_audit
import step1_ema_bn_s42 as ema_mod
import step1_box_fusion_replay_s42 as replay_mod


def fmt_native_pct(res: dict) -> dict:
    nm = res["native_metrics"]
    return dict(
        mAP50=round(float(nm["metrics/mAP50(B)"]) * 100.0, 3),
        AP75=round(float(nm["metrics/AP75(B)"]) * 100.0, 3),
        mAP50_95=round(float(nm["metrics/mAP50-95(B)"]) * 100.0, 3),
        P=round(float(nm["metrics/precision(B)"]) * 100.0, 3),
        Recall=round(float(nm["metrics/recall(B)"]) * 100.0, 3),
    )


def eval_b1_ckpt(model: torch.nn.Module, ckpt_out: Path, eval_dir: Path) -> tuple[dict, dict]:
    ckpt_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(model=model.cpu().float(), ema=None, epoch=-1, optimizer=None), ckpt_out)
    if eval_dir.exists():
        shutil.rmtree(eval_dir)
    res = reference.evaluate_checkpoint(ckpt_out, eval_dir, run_v2.DATA_YAML)
    slices = run_v6.compute_extended_slices(eval_dir / "predictions.coco.json")
    return res, slices


def cluster_image_nms_exact(img_rec: dict, conf_thres: float = 0.001, iou_thres: float = 0.70, max_det: int = 300) -> list[dict]:
    """Exact replica of DetectionValidator's CUDA torchvision.ops.nms (+c*7680 offset) for cluster heads."""
    b640 = img_rec["boxes_640"]
    b1024 = img_rec["boxes_1024"]
    sc = img_rec["scores"]
    lv = img_rec["levels"]

    a_idx_np, c_idx_np = np.nonzero(sc > conf_thres)
    if len(a_idx_np) == 0:
        return []
    cand_scores_np = sc[a_idx_np, c_idx_np]
    if len(cand_scores_np) > 30000:
        cand_t = torch.from_numpy(cand_scores_np).cuda()
        top_ord = torch.argsort(cand_t, descending=True)[:30000].cpu().numpy()
        a_idx_np, c_idx_np, cand_scores_np = a_idx_np[top_ord], c_idx_np[top_ord], cand_scores_np[top_ord]

    max_wh = 7680.0
    b640_off_np = b640[a_idx_np] + (c_idx_np.astype(np.float32) * max_wh)[:, None]
    kept_tv = torchvision.ops.nms(
        torch.from_numpy(b640_off_np).cuda(),
        torch.from_numpy(cand_scores_np).cuda(),
        iou_thres,
    ).cpu().numpy()[:max_det]

    kept_rank = {int(pos): rank for rank, pos in enumerate(kept_tv)}
    kept_set = set(kept_rank.keys())

    # Group all candidates by class sorted descending by score (matching torchvision.ops.nms sort)
    by_class = defaultdict(list)
    for pos in np.argsort(-cand_scores_np, kind="mergesort"):
        by_class[int(c_idx_np[pos])].append(int(pos))

    clusters_by_rank: dict[int, dict] = {}
    for c, pos_list in by_class.items():
        pos_arr = np.array(pos_list, dtype=np.int64)
        anch = a_idx_np[pos_arr]
        c_b640 = b640[anch]
        c_b640_off = b640_off_np[pos_arr]
        c_b1024 = b1024[anch]
        c_sc = cand_scores_np[pos_arr]
        c_lv = lv[anch]

        surv_local = [idx for idx, gp in enumerate(pos_arr) if int(gp) in kept_set]
        if not surv_local:
            continue
        surv_set = set(surv_local)
        iou_to_surv_off = replay_mod.iou_matrix_xyxy(c_b640_off, c_b640_off[surv_local])
        iou_to_surv_raw = replay_mod.iou_matrix_xyxy(c_b640, c_b640[surv_local])

        sup_rank_for_pos = np.full(len(pos_arr), -1, dtype=np.int64)
        for idx in range(len(pos_arr)):
            if idx in surv_set:
                sup_rank_for_pos[idx] = surv_local.index(idx)
                continue
            higher_hits = [sr for sr, s_idx in enumerate(surv_local) if s_idx < idx and iou_to_surv_off[idx, sr] > iou_thres]
            if higher_hits:
                sup_rank_for_pos[idx] = higher_hits[0]
            else:
                higher_all = [sr for sr, s_idx in enumerate(surv_local) if s_idx < idx]
                if higher_all:
                    best_sr = max(higher_all, key=lambda sr: iou_to_surv_raw[idx, sr])
                    if iou_to_surv_raw[idx, best_sr] > iou_thres - 1e-3:
                        sup_rank_for_pos[idx] = best_sr

        best_sr_all = np.argmax(iou_to_surv_raw, axis=1)
        best_iou_all = iou_to_surv_raw[np.arange(len(pos_arr)), best_sr_all]
        is_suppressed = np.array([k not in surv_set for k in range(len(pos_arr))], dtype=bool)

        for sr, s_idx in enumerate(surv_local):
            g_pos = int(pos_arr[s_idx])
            c70_idx = np.flatnonzero(sup_rank_for_pos == sr)
            extra55_idx = np.flatnonzero((np.arange(len(pos_arr)) == s_idx) | (is_suppressed & (best_sr_all == sr) & (best_iou_all >= 0.55)))
            clusters_by_rank[kept_rank[g_pos]] = dict(
                class_idx=c,
                anchor_score=float(c_sc[s_idx]),
                anchor_level=int(c_lv[s_idx]),
                anchor_box_1024=c_b1024[s_idx].copy(),
                anchor_box_640=c_b640[s_idx].copy(),
                c70_boxes_1024=c_b1024[c70_idx],
                c70_boxes_640=c_b640[c70_idx],
                c70_scores=c_sc[c70_idx],
                c70_levels=c_lv[c70_idx],
                c55_boxes_1024=c_b1024[extra55_idx],
                c55_scores=c_sc[extra55_idx],
            )

    return [clusters_by_rank[r] for r in range(len(kept_tv)) if r in clusters_by_rank]


def main():
    arm = "b1"
    gt = COCO(str(cpu_audit.ANN_VAL))
    assert len(gt.anns) >= 2612
    loader = ema_mod.build_clean_train_loader(batch=32)

    tmp_root = STAGE / "diagnostics/_tmp_b1_ema"
    if tmp_root.exists():
        shutil.rmtree(tmp_root)
    tmp_root.mkdir(parents=True, exist_ok=True)

    out = dict(
        generated_at=dt.datetime.now().isoformat(),
        arm=arm,
        seed=42,
        snapshot_indexing_note=(
            "Ultralytics save_period=5 writes 1-indexed Epoch k+1 as epoch{k}.pt: "
            "epoch60..85 = E61..E86 (6 ckpts), epoch65..90 = E66..E91 (6 ckpts), epoch60..90 = E61..E91 (7 ckpts)."
        ),
        protocol=(
            "SVRDD7 Val 1,000 imgs / 2,612 GTs, FP32 O2M + class-aware NMS iou=0.70, "
            "COCO maxDets=100; cumulative 1-pass BN recalibration on 6,000 clean Train imgs."
        ),
        windows={},
        box_fusion_replay={},
    )

    # Fail-fast: run exact replay on b1_best_E80 first (uses CUDA torchvision.ops.nms survivors)
    replay_mod.cluster_image_nms = cluster_image_nms_exact
    b1_best_ck = STAGE / "runs/b1/weights/best.pt"
    b1_best_mj = STAGE / "records/b1/evaluation/best/metrics.json"
    print("=== Running Fail-Fast Box Voting Replay on: b1_best_E80 ===", flush=True)
    rep_best = replay_mod.replay_checkpoint(b1_best_ck, "b1_best_E80", b1_best_mj)
    rep_best["unmatched75_note"] = (
        "WARNING: step1_box_fusion_replay_s42.capture_raw_candidates pre-filters anchors with sc.max(axis=1) > 0.001 (line 85). "
        "Do NOT claim its unmatched_raw_geometry_miss as all-34,000-anchor unfiltered miss; refer to diagnostics/b1/summary.json for true all-anchor counts."
    )
    out["box_fusion_replay"]["b1_best_E80"] = rep_best

    candidates_for_replay = []

    print(f"=== V6 Zero-Retrain EMA + Clean BN Recalibration: {arm} ===", flush=True)
    for wname, epochs in ema_mod.WINDOWS.items():
        pre_model, avg_meta = ema_mod.average_checkpoints(arm, epochs)
        avg_meta["epoch_filenames"] = [f"epoch{ep}.pt" for ep in epochs]
        avg_meta["one_indexed_epochs"] = [ep + 1 for ep in epochs]

        pre_ck = STAGE / "runs" / arm / "weights" / f"ema_{wname}_pre.pt"
        pre_eval = tmp_root / f"eval_{wname}_pre"
        pre_res, pre_slices = eval_b1_ckpt(pre_model, pre_ck, pre_eval)
        pre_main = ema_mod.fmt_main_pct(pre_res)
        pre_nat = fmt_native_pct(pre_res)

        recal_model, bn_meta = ema_mod.recalibrate_bn_exact(pre_model, loader)
        assert bn_meta["bn_layers_recalibrated"] == 152, f"Expected 152 BN layers for b1, got {bn_meta}"
        shallow_bns = [
            (n, m)
            for n, m in recal_model.named_modules()
            if isinstance(m, torch.nn.BatchNorm2d) and "shallow_side" in n
        ]
        assert len(shallow_bns) == 2 and all(
            int(m.num_batches_tracked.item()) == bn_meta["batches"] for _, m in shallow_bns
        ), f"shallow_side BN recal check failed: {shallow_bns}"
        bn_meta["shallow_side_bn_tracked"] = len(shallow_bns)

        post_ck = STAGE / "runs" / arm / "weights" / f"ema_{wname}_bn_recal.pt"
        post_eval = tmp_root / f"eval_{wname}_post"
        post_res, post_slices = eval_b1_ckpt(recal_model, post_ck, post_eval)
        post_main = ema_mod.fmt_main_pct(post_res)
        post_nat = fmt_native_pct(post_res)

        delta = {
            k: round(post_main[k] - pre_main[k], 3)
            for k in ("AP50_95", "AP50", "AP75", "AP_small", "AP_medium", "AP_large", "AR100")
        }

        print(
            f"  [b1 {wname}] PRE : 95={pre_main['AP50_95']:.3f} 50={pre_main['AP50']:.3f} 75={pre_main['AP75']:.3f} "
            f"S={pre_main['AP_small']:.3f} lt8={pre_slices['short640_lt8']['AP50_95']:.3f} | nat={pre_nat['mAP50']:.3f}/{pre_nat['mAP50_95']:.3f}",
            flush=True,
        )
        print(
            f"  [b1 {wname}] POST: 95={post_main['AP50_95']:.3f} ({delta['AP50_95']:+.3f}) "
            f"50={post_main['AP50']:.3f} ({delta['AP50']:+.3f}) 75={post_main['AP75']:.3f} ({delta['AP75']:+.3f}) "
            f"S={post_main['AP_small']:.3f} lt8={post_slices['short640_lt8']['AP50_95']:.3f} | nat={post_nat['mAP50']:.3f}/{post_nat['mAP50_95']:.3f}",
            flush=True,
        )

        out["windows"][wname] = dict(
            averaging=avg_meta,
            bn_recalibration=bn_meta,
            pre_checkpoint=str(pre_ck),
            pre_sha256=pre_res["sha256"],
            pre_recal=dict(main=pre_main, native_metrics=pre_nat, slices=pre_slices, classes=pre_res["classes"]),
            recalibrated_checkpoint=str(post_ck),
            recalibrated_sha256=post_res["sha256"],
            post_recal=dict(main=post_main, native_metrics=post_nat, slices=post_slices, classes=post_res["classes"]),
            delta_post_minus_pre=delta,
        )

        candidates_for_replay.append(
            (pre_main["AP50_95"], f"b1_ema_{wname}_pre", pre_ck, pre_eval / "metrics.json")
        )
        candidates_for_replay.append(
            (post_main["AP50_95"], f"b1_ema_{wname}_bn_recal", post_ck, post_eval / "metrics.json")
        )

    # Sort averaged checkpoints descending by COCO AP50:95
    candidates_for_replay.sort(key=lambda t: t[0], reverse=True)
    top_variant = candidates_for_replay[0]
    e6085_pre_variant = next(t for t in candidates_for_replay if t[1] == "b1_ema_E60_85_pre")

    replay_targets = [
        (top_variant[1], top_variant[2], top_variant[3]),
    ]
    if e6085_pre_variant[1] != top_variant[1]:
        replay_targets.append((e6085_pre_variant[1], e6085_pre_variant[2], e6085_pre_variant[3]))
    # If top_variant is PRE, also evaluate top POST-BN variant (and vice-versa) so both PRE and POST voting are directly observed
    if top_variant[1].endswith("_pre"):
        top_post = next(t for t in candidates_for_replay if t[1].endswith("_bn_recal"))
        if top_post[1] not in {x[0] for x in replay_targets}:
            replay_targets.append((top_post[1], top_post[2], top_post[3]))

    out["selected_replay_targets"] = ["b1_best_E80"] + [t[0] for t in replay_targets]
    print(f"=== Running Box Voting Replay on averaged targets: {[t[0] for t in replay_targets]} ===", flush=True)

    for tag, ck_path, mj_path in replay_targets:
        rep = replay_mod.replay_checkpoint(ck_path, tag, mj_path)
        rep["unmatched75_note"] = (
            "WARNING: step1_box_fusion_replay_s42.capture_raw_candidates pre-filters anchors with sc.max(axis=1) > 0.001 (line 85). "
            "Do NOT claim its unmatched_raw_geometry_miss as all-34,000-anchor unfiltered miss; refer to diagnostics/b1/summary.json for true all-anchor counts."
        )
        out["box_fusion_replay"][tag] = rep

    out_path = STAGE / "diagnostics/b1_ema6_bn_s42.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"[DONE] Saved {out_path}", flush=True)

    shutil.rmtree(tmp_root, ignore_errors=True)
    shutil.rmtree(STAGE / "diagnostics/_tmp_raw_replay", ignore_errors=True)


if __name__ == "__main__":
    main()
