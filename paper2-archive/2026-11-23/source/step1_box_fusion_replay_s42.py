"""Step 1-2: Score-Preserving Intra-Cluster Box Selection & Fusion Offline Replay (Seed 42).

- Captures ALL raw pre-NMS O2M candidates (score > 0.001) on SVRDD7 Val (1,000 imgs / 2,612 GTs)
  using exact TaskValidator / Audit preprocessing & scale_boxes conventions.
- Verifies strict native greedy class-aware NMS (iou=0.70, conf=0.001, max_det=300) parity
  against official metrics.json (assert max_abs_diff <= 0.005 pp).
- Evaluates strictly score-preserving intra-cluster box coordinate replacement / fusion methods
  plus diagnostic-only intra-cluster Oracle ceiling and unmatched IoU>=0.75 GT attribution.
"""
import argparse
import contextlib
import io
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torchvision
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

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
from ultralytics.utils import ops

NAMES = cpu_audit.NAMES
ANN_VAL = cpu_audit.ANN_VAL
GAIN_640 = cpu_audit.GAIN_640


def iou_matrix_xyxy(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    lo = np.maximum(a[:, None, :2], b[None, :, :2])
    hi = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.maximum(hi - lo, 0.0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    area_a = np.maximum((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), 0.0)
    area_b = np.maximum((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]), 0.0)
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-12)


class RawCaptureValidator(reference.TaskValidator):
    """Captures exact raw O2M decoded boxes (640 + 1024) and 7-class scores before NMS."""

    def postprocess(self, preds):
        decoded, meta = preds
        assert not self.end2end and decoded.shape[1] == 11 and len(meta["feats"]) == 4
        y = decoded.detach().clone()
        levels = np.concatenate(
            [np.full(int(v.shape[-2] * v.shape[-1]), k, np.int8) for k, v in enumerate(meta["feats"], 2)]
        )
        assert len(levels) == y.shape[2]
        boxes_640 = ops.xywh2xyxy(y[:, :4].transpose(1, 2))
        scores_all = y[:, 4:].transpose(1, 2)
        self._raw_batch = (boxes_640, scores_all, levels)
        return super().postprocess(preds)

    def update_metrics(self, preds, batch):
        boxes_640, scores_all, levels = self._raw_batch
        for si, name in enumerate(batch["im_file"]):
            pb = self._prepare_batch(si, batch)
            image_id = self.image_ids[Path(name).stem]
            b640 = boxes_640[si].cpu().numpy().astype(np.float32)
            b1024 = ops.scale_boxes(pb["imgsz"], boxes_640[si].clone(), pb["ori_shape"], pb["ratio_pad"]).cpu().numpy().astype(np.float32)
            sc = scores_all[si].cpu().numpy().astype(np.float32)
            keep_mask = sc.max(axis=1) > 0.001
            self.captured_images[image_id] = dict(
                image_id=int(image_id),
                fname=Path(name).name,
                boxes_640=b640[keep_mask],
                boxes_1024=b1024[keep_mask],
                scores=sc[keep_mask],
                levels=levels[keep_mask],
            )
        self._raw_batch = None
        super().update_metrics(preds, batch)


def capture_raw_candidates(ckpt_path: Path, data_yaml: Path) -> tuple[dict, dict, list[dict]]:
    model = reference.load_model(ckpt_path)
    v = RawCaptureValidator(
        args=dict(
            task="detect",
            data=str(data_yaml),
            imgsz=640,
            batch=32,
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
        save_dir=STAGE / "diagnostics" / "_tmp_raw_replay",
    )
    v.captured_images = {}
    gt = COCO(str(ANN_VAL))
    v.ground_truth = gt
    v.image_ids = {Path(im["file_name"]).stem: k for k, im in gt.imgs.items()}
    v.categories = {c["name"]: k for k, c in gt.cats.items()}
    native = v(model=model)
    assert len(v.captured_images) == 1000
    official_preds = []
    for p in v.jdict:
        idx = int(p["category_id"]) - 1
        official_preds.append(
            dict(
                image_id=v.image_ids[Path(p["file_name"]).stem],
                category_id=v.categories[reference.NAMES[idx]],
                bbox=p["bbox"],
                score=p["score"],
            )
        )
    return v.captured_images, dict(gt=gt, categories=v.categories, native=native), official_preds


def cluster_image_nms(img_rec: dict, conf_thres: float = 0.001, iou_thres: float = 0.70, max_det: int = 300):
    """Exact greedy class-aware NMS reproducing Ultralytics multi_label=True + max_det=300.

    Returns list of cluster dicts sorted globally descending by anchor score ([:max_det]),
    preserving exact surviving class_idx, anchor_score, and order across all fusion methods.
    """
    b640 = img_rec["boxes_640"]
    b1024 = img_rec["boxes_1024"]
    sc = img_rec["scores"]
    lv = img_rec["levels"]

    # Gather all (anchor_idx, class_idx) with score > conf_thres, just like multi_label=True
    a_idx, c_idx = np.nonzero(sc > conf_thres)
    if len(a_idx) == 0:
        return []
    cand_scores = sc[a_idx, c_idx]
    # Ultralytics limits pre-NMS candidates to max_nms=30000 by confidence
    order = np.argsort(-cand_scores, kind="mergesort")
    if len(order) > 30000:
        order = order[:30000]
    a_idx = a_idx[order]
    c_idx = c_idx[order]
    cand_scores = cand_scores[order]

    # Run torchvision batched/class-offset NMS on 640 coordinates to verify exact survivor set
    max_wh = 7680.0
    c_offset = c_idx.astype(np.float32) * max_wh
    boxes_off = torch.from_numpy(b640[a_idx] + c_offset[:, None])
    scores_t = torch.from_numpy(cand_scores)
    kept_tv = torchvision.ops.nms(boxes_off, scores_t, iou_thres).numpy()

    # Build exact suppression clusters per class among those candidates
    clusters = []
    # Map candidate position -> cluster
    by_class = defaultdict(list)
    for pos in range(len(a_idx)):
        by_class[int(c_idx[pos])].append(pos)

    kept_set = set(int(x) for x in kept_tv)
    kept_clusters_by_pos = {}

    for c, pos_list in by_class.items():
        pos_arr = np.array(pos_list, dtype=np.int64)  # already sorted descending by cand_scores!
        anch = a_idx[pos_arr]
        c_b640 = b640[anch]
        c_b1024 = b1024[anch]
        c_sc = cand_scores[pos_arr]
        c_lv = lv[anch]
        iou_mat = iou_matrix_xyxy(c_b640, c_b640)

        suppressed_by = [-1] * len(pos_arr)
        survivor_local_idxs = []
        for i in range(len(pos_arr)):
            if suppressed_by[i] != -1:
                continue
            survivor_local_idxs.append(i)
            suppressed_by[i] = i
            # Any lower-score remaining candidate with IoU > iou_thres gets suppressed by i
            hits = np.flatnonzero((iou_mat[i, i + 1 :] > iou_thres)) + (i + 1)
            for j in hits:
                if suppressed_by[j] == -1:
                    suppressed_by[j] = i

        # Check greedy vs torchvision NMS consistency
        local_kept_pos = set(int(pos_arr[i]) for i in survivor_local_idxs)
        tv_c_pos = set(int(p) for p in pos_arr if int(p) in kept_set)
        # Note: exact ties in float32 can differ trivially if any, usually 100% equal

        # Assign each non-survivor j to its greedy suppressor i (C70), and precompute best kept IoU >= 0.55
        sup_arr = np.array(suppressed_by, dtype=np.int64)
        surv_b640 = c_b640[survivor_local_idxs]
        surv_set = set(survivor_local_idxs)
        is_suppressed = np.array([k not in surv_set for k in range(len(pos_arr))], dtype=bool)
        if len(survivor_local_idxs):
            iou_to_surv = iou_matrix_xyxy(c_b640, surv_b640)
            best_s = np.argmax(iou_to_surv, axis=1)
            best_iou = iou_to_surv[np.arange(len(pos_arr)), best_s]
        else:
            best_s = np.zeros((len(pos_arr),), dtype=np.int64)
            best_iou = np.zeros((len(pos_arr),), dtype=np.float32)

        for s_rank, i in enumerate(survivor_local_idxs):
            c70_idx = np.flatnonzero(sup_arr == i)
            if len(survivor_local_idxs):
                extra55_idx = np.flatnonzero((np.arange(len(pos_arr)) == i) | (is_suppressed & (best_s == s_rank) & (best_iou >= 0.55)))
            else:
                extra55_idx = np.array([i])

            g_pos = int(pos_arr[i])
            kept_clusters_by_pos[g_pos] = dict(
                class_idx=c,
                anchor_score=float(c_sc[i]),
                anchor_level=int(c_lv[i]),
                anchor_box_1024=c_b1024[i].copy(),
                anchor_box_640=c_b640[i].copy(),
                c70_boxes_1024=c_b1024[c70_idx],
                c70_boxes_640=c_b640[c70_idx],
                c70_scores=c_sc[c70_idx],
                c70_levels=c_lv[c70_idx],
                c55_boxes_1024=c_b1024[extra55_idx],
                c55_scores=c_sc[extra55_idx],
            )

    # Emit exact top max_det=300 ordered by global descending anchor_score
    for g_pos in sorted(kept_clusters_by_pos.keys())[:max_det]:
        clusters.append(kept_clusters_by_pos[g_pos])
    return clusters


def resolve_box_variants(clusters: list[dict], gt_boxes_1024_by_cls: dict[int, np.ndarray], method: str) -> list[dict]:
    out = []
    for cl in clusters:
        c = cl["class_idx"]
        b_nat = cl["anchor_box_1024"]
        sc70 = cl["c70_scores"]
        b70 = cl["c70_boxes_1024"]
        b70_640 = cl["c70_boxes_640"]

        if method == "native_nms" or (len(b70) == 1 and method != "box_voting_p2_t55"):
            box = b_nat
        elif method == "consensus_iou":
            imat = iou_matrix_xyxy(b70_640, b70_640)
            consensus = imat @ sc70
            best_m = int(np.argmax(consensus))
            box = b70[best_m]
        elif method == "box_voting_p1_t70":
            w = sc70[:, None]
            box = (b70 * w).sum(axis=0) / w.sum()
        elif method == "box_voting_p2_t70":
            w = (sc70 ** 2)[:, None]
            box = (b70 * w).sum(axis=0) / w.sum()
        elif method == "box_voting_p2_t80":
            ious_to_anchor = iou_matrix_xyxy(b70_640[0:1], b70_640)[0]
            mask = ious_to_anchor >= 0.80
            if not np.any(mask):
                mask[0] = True
            w = (sc70[mask] ** 2)[:, None]
            box = (b70[mask] * w).sum(axis=0) / w.sum()
        elif method == "box_voting_p2_t55":
            b55 = cl["c55_boxes_1024"]
            sc55 = cl["c55_scores"]
            w = (sc55 ** 2)[:, None]
            box = (b55 * w).sum(axis=0) / w.sum()
        elif method == "cluster_oracle_diagnostic_only":
            gts = gt_boxes_1024_by_cls.get(c)
            if gts is not None and len(gts):
                imat = iou_matrix_xyxy(b70, gts)
                best_per_cand = imat.max(axis=1)
                if best_per_cand.max() > 0.0:
                    box = b70[int(np.argmax(best_per_cand))]
                else:
                    box = b_nat
            else:
                box = b_nat
        else:
            raise ValueError(method)

        x1, y1, x2, y2 = [float(v) for v in box]
        out.append(dict(class_idx=c, score=cl["anchor_score"], bbox_xyxy=box, bbox_coco=[x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)]))
    return out


def evaluate_prediction_set(
    gt: COCO,
    cats_by_name: dict[str, int],
    preds_coco: list[dict],
    by_img_preds_xyxy: dict[int, list[dict]],
    gt_boxes_by_img: dict[int, list[tuple[int, np.ndarray]]],
) -> dict:
    dt_coco = cpu_audit.build_coco_dt(gt, preds_coco)
    all_cats = [cats_by_name[n] for n in NAMES]
    ev = cpu_audit.official_eval(gt, dt_coco, all_cats)
    pr = ev.eval["precision"]
    rc = ev.eval["recall"]
    main = dict(
        AP50=round(float(cpu_audit.extract_coco_metrics(ev, 0)["AP50"] * 100), 3),
        AP75=round(float(cpu_audit.extract_coco_metrics(ev, 0)["AP75"] * 100), 3),
        AP50_95=round(float(cpu_audit.extract_coco_metrics(ev, 0)["AP50_95"] * 100), 3),
        AP_small=round(float(cpu_audit.extract_coco_metrics(ev, 1)["AP50_95"] * 100), 3),
        AP_medium=round(float(cpu_audit.extract_coco_metrics(ev, 2)["AP50_95"] * 100), 3),
        AP_large=round(float(cpu_audit.extract_coco_metrics(ev, 3)["AP50_95"] * 100), 3),
        AR100=round(float(cpu_audit.extract_coco_metrics(ev, 0)["AR100"] * 100), 3),
    )

    slices = cpu_audit.evaluate_all_slices(gt, dt_coco, cats_by_name)
    per_class = {}
    for ci, cname in enumerate(NAMES):
        c_all = pr[:, :, ci, 0, -1]
        c_50 = pr[0, :, ci, 0, -1]
        c_75 = pr[5, :, ci, 0, -1]
        per_class[cname] = dict(
            AP50=round(float(c_50[c_50 > -1].mean() * 100), 2),
            AP75=round(float(c_75[c_75 > -1].mean() * 100), 2),
            AP50_95=round(float(c_all[c_all > -1].mean() * 100), 2),
        )

    # Diagnostic counts @ 0.25: bg_FP, duplicates on same GT, cross-target merges
    det_at025 = 0
    bg_fp_same = 0
    bg_fp_any = 0
    duplicate_extra = 0
    cross_target_merge = 0
    matched75_official = 0
    matched75_at025 = 0

    for item in ev.evalImgs:
        if item is None or list(item["aRng"]) != list(ev.params.areaRng[0]):
            continue
        k75 = int(np.argmin(np.abs(ev.params.iouThrs - 0.75)))
        for j, gid in enumerate(item["gtIds"]):
            if item["gtIgnore"][j]:
                continue
            m_id = int(item["gtMatches"][k75, j])
            if m_id > 0:
                matched75_official += 1
                if dt_coco.anns[m_id]["score"] >= 0.25:
                    matched75_at025 += 1

    for img_id, plist in by_img_preds_xyxy.items():
        gts = gt_boxes_by_img.get(img_id, [])
        p25 = [p for p in plist if p["score"] >= 0.25]
        det_at025 += len(p25)
        if not p25:
            continue
        if not gts:
            bg_fp_same += len(p25)
            bg_fp_any += len(p25)
            continue

        gt_by_c = defaultdict(list)
        all_gt_boxes = np.stack([g[1] for g in gts], axis=0)
        for gc, gb in gts:
            gt_by_c[gc].append(gb)

        gt_hit_counts = {gc: np.zeros(len(arr), dtype=np.int32) for gc, arr in gt_by_c.items()}
        for p in p25:
            pb = p["bbox_xyxy"][None, :]
            max_any = float(iou_matrix_xyxy(pb, all_gt_boxes)[0].max())
            if max_any < 0.10:
                bg_fp_any += 1
            same_list = gt_by_c.get(p["class_idx"])
            if not same_list:
                bg_fp_same += 1
                continue
            same_arr = np.stack(same_list, axis=0)
            ious_s = iou_matrix_xyxy(pb, same_arr)[0]
            if float(ious_s.max()) < 0.10:
                bg_fp_same += 1
            best_g = int(np.argmax(ious_s))
            if ious_s[best_g] >= 0.50:
                gt_hit_counts[p["class_idx"]][best_g] += 1

            # Check cross-target merge: detection overlaps >=2 distinct same-class GTs with IoU>=0.10 or covers >=50% of >=2 GTs
            if len(same_arr) >= 2:
                lo = np.maximum(same_arr[:, :2], pb[0, :2])
                hi = np.minimum(same_arr[:, 2:], pb[0, 2:])
                wh = np.maximum(hi - lo, 0.0)
                inter = wh[:, 0] * wh[:, 1]
                gt_area = np.maximum((same_arr[:, 2] - same_arr[:, 0]) * (same_arr[:, 3] - same_arr[:, 1]), 1.0)
                cov = inter / gt_area
                if int(((ious_s >= 0.10) | (cov >= 0.50)).sum()) >= 2:
                    cross_target_merge += 1

        for cnts in gt_hit_counts.values():
            duplicate_extra += int(np.maximum(cnts - 1, 0).sum())

    return dict(
        main=main,
        slices=slices,
        classes=per_class,
        diagnostics_at025=dict(
            det_at025=det_at025,
            bg_fp_iou_any_lt01=bg_fp_any,
            bg_fp_iou_same_lt01=bg_fp_same,
            duplicate_extra_hits_ge050=duplicate_extra,
            cross_target_merge_ge2_gts=cross_target_merge,
            matched75_official=matched75_official,
            matched75_at025=matched75_at025,
        ),
    )


def analyze_unmatched75_breakdown(
    gt: COCO,
    cats_by_name: dict[str, int],
    raw_by_img: dict,
    clusters_by_img: dict,
    native_preds_coco: list[dict],
) -> dict:
    """Exact partition of all 2,612 Val GTs unmatched at IoU=0.75 (official & @0.25)."""
    dt_coco = cpu_audit.build_coco_dt(gt, native_preds_coco)
    ev = cpu_audit.official_eval(gt, dt_coco, [cats_by_name[n] for n in NAMES])
    k75 = int(np.argmin(np.abs(ev.params.iouThrs - 0.75)))
    match_score_75 = {}
    for item in ev.evalImgs:
        if item is None or list(item["aRng"]) != list(ev.params.areaRng[0]):
            continue
        for j, gid in enumerate(item["gtIds"]):
            if item["gtIgnore"][j]:
                continue
            mid = int(item["gtMatches"][k75, j])
            match_score_75[int(gid)] = float(dt_coco.anns[mid]["score"]) if mid > 0 else None

    cname_to_idx = {n: i for i, n in enumerate(NAMES)}
    cat_id_to_idx = {cats_by_name[n]: cname_to_idx[n] for n in NAMES}

    cats_counts = Counter()
    cats_at025 = Counter()
    for aid, ann in gt.anns.items():
        if ann.get("iscrowd") or ann.get("ignore"):
            continue
        img_id = ann["image_id"]
        cidx = cat_id_to_idx[ann["category_id"]]
        x, y, w, h = ann["bbox"]
        gb = np.array([[x, y, x + w, y + h]], dtype=np.float32)

        ms = match_score_75.get(aid)
        if ms is not None:
            cats_counts["matched_official"] += 1
            if ms >= 0.25:
                cats_at025["matched_at025"] += 1
                continue

        rec = raw_by_img[img_id]
        b1024 = rec["boxes_1024"]
        sc_c = rec["scores"][:, cidx]
        ious = iou_matrix_xyxy(gb, b1024)[0] if len(b1024) else np.zeros((0,), dtype=np.float32)
        has_raw75 = bool(np.any(ious >= 0.75))
        has_floor75 = bool(np.any((ious >= 0.75) & (sc_c > 0.001)))

        if ms is None:
            if not has_raw75:
                cats_counts["unmatched_raw_geometry_miss"] += 1
            elif not has_floor75:
                cats_counts["unmatched_score_below_0001"] += 1
            else:
                # Check if inside any surviving C70 cluster of this class in top-300
                cl_c = [cl for cl in clusters_by_img[img_id] if cl["class_idx"] == cidx]
                in_c70_anchor_ge75 = any(float(iou_matrix_xyxy(gb, cl["anchor_box_1024"][None, :])[0, 0]) >= 0.75 for cl in cl_c)
                in_c70_member_ge75 = any(float(iou_matrix_xyxy(gb, cl["c70_boxes_1024"])[0].max()) >= 0.75 for cl in cl_c)
                if in_c70_member_ge75 and not in_c70_anchor_ge75:
                    cats_counts["unmatched_inside_surviving_c70_cluster_suppressed_by_coarse_anchor"] += 1
                elif in_c70_anchor_ge75:
                    cats_counts["unmatched_anchor_has75_but_stolen_by_neighbor_gt_or_cap100"] += 1
                else:
                    cats_counts["unmatched_outside_c70_top300_cap_or_wrong_class"] += 1

        # Breakdown at @0.25 working threshold too
        if ms is None or ms < 0.25:
            if not has_raw75:
                cats_at025["miss025_raw_geometry_miss"] += 1
            elif not has_floor75:
                cats_at025["miss025_score_below_0001"] += 1
            else:
                cl_c = [cl for cl in clusters_by_img[img_id] if cl["class_idx"] == cidx]
                cl_c25 = [cl for cl in cl_c if cl["anchor_score"] >= 0.25]
                in_c70_25_mem = any(float(iou_matrix_xyxy(gb, cl["c70_boxes_1024"])[0].max()) >= 0.75 for cl in cl_c25)
                in_c70_low = any(float(iou_matrix_xyxy(gb, cl["c70_boxes_1024"])[0].max()) >= 0.75 for cl in cl_c if cl["anchor_score"] < 0.25)
                if in_c70_25_mem:
                    cats_at025["miss025_inside_ge025_c70_cluster"] += 1
                elif in_c70_low:
                    cats_at025["miss025_anchor_score_0001_to_025"] += 1
                else:
                    cats_at025["miss025_outside_c70_cap300"] += 1

    return dict(official_coco=dict(cats_counts), at_025=dict(cats_at025))


def replay_checkpoint(ckpt_path: Path, tag: str, official_metrics_json: Path | None = None) -> dict:
    raw_by_img, meta, official_preds = capture_raw_candidates(ckpt_path, run_v2.DATA_YAML)
    gt = meta["gt"]
    cats_by_name = meta["categories"]
    cat_ids_list = [cats_by_name[n] for n in NAMES]

    # Ground truths organized by image
    cname_to_idx = {n: i for i, n in enumerate(NAMES)}
    cat_id_to_idx = {cats_by_name[n]: cname_to_idx[n] for n in NAMES}
    gt_boxes_by_img = defaultdict(list)
    gt_boxes_1024_by_img_cls = defaultdict(lambda: defaultdict(list))
    for ann in gt.anns.values():
        if ann.get("iscrowd") or ann.get("ignore"):
            continue
        x, y, w, h = ann["bbox"]
        box = np.array([x, y, x + w, y + h], dtype=np.float32)
        cidx = cat_id_to_idx[ann["category_id"]]
        gt_boxes_by_img[ann["image_id"]].append((cidx, box))
        gt_boxes_1024_by_img_cls[ann["image_id"]][cidx].append(box)
    gt_np = {
        img_id: {c: np.stack(arr, axis=0) for c, arr in cls_d.items()}
        for img_id, cls_d in gt_boxes_1024_by_img_cls.items()
    }

    # Build clusters once
    clusters_by_img = {img_id: cluster_image_nms(rec) for img_id, rec in raw_by_img.items()}

    # Official parity check from Validator output vs metrics.json
    ev_off = cpu_audit.official_eval(gt, cpu_audit.build_coco_dt(gt, official_preds), cat_ids_list)
    off_m = {k: round(v * 100, 4) for k, v in cpu_audit.extract_coco_metrics(ev_off, 0).items()}

    methods = (
        "native_nms",
        "consensus_iou",
        "box_voting_p1_t70",
        "box_voting_p2_t70",
        "box_voting_p2_t80",
        "box_voting_p2_t55",
        "cluster_oracle_diagnostic_only",
    )

    method_results = {}
    native_coco_preds = None
    for mname in methods:
        coco_preds = []
        by_img_xyxy = {}
        for img_id, cl_list in clusters_by_img.items():
            resolved = resolve_box_variants(cl_list, gt_np.get(img_id, {}), mname)
            by_img_xyxy[img_id] = resolved
            for r in resolved:
                coco_preds.append(
                    dict(
                        image_id=int(img_id),
                        category_id=cat_ids_list[r["class_idx"]],
                        bbox=[round(float(v), 3) for v in r["bbox_coco"]],
                        score=round(float(r["score"]), 5),
                    )
                )
        res = evaluate_prediction_set(gt, cats_by_name, coco_preds, by_img_xyxy, gt_boxes_by_img)
        method_results[mname] = res
        if mname == "native_nms":
            native_coco_preds = coco_preds
            # Parity gate!
            diff_vs_validator = max(
                abs(res["main"][k] - off_m[k]) for k in ("AP50_95", "AP50", "AP75")
            )
            assert diff_vs_validator <= 0.005, f"Validator NMS parity failed: {res['main']} vs {off_m}"
            if official_metrics_json is not None and official_metrics_json.is_file():
                ref_m = json.loads(official_metrics_json.read_text())["main"]
                diff_vs_official = max(
                    abs(res["main"][k] - ref_m[k] * 100.0) for k in ("AP50_95", "AP50", "AP75")
                )
                assert diff_vs_official <= 0.005, f"Official metrics.json parity failed: {res['main']} vs {ref_m}"
                parity_diff = diff_vs_official
            else:
                parity_diff = diff_vs_validator
            print(f"[Parity OK: {tag}] max_diff={parity_diff:.4f} pp | native={res['main']}", flush=True)

        m = res["main"]
        s = res["slices"]
        d025 = res["diagnostics_at025"]
        print(
            f"  [{tag}/{mname}] 95={m['AP50_95']:.3f} 50={m['AP50']:.3f} 75={m['AP75']:.3f} AR={m['AR100']:.3f} | "
            f"lt8={s['short640_lt8']['AP50_95']:.3f} | bgFP={d025['bg_fp_iou_any_lt01']} dup={d025['duplicate_extra_hits_ge050']} merge={d025['cross_target_merge_ge2_gts']}",
            flush=True,
        )

    breakdown = analyze_unmatched75_breakdown(gt, cats_by_name, raw_by_img, clusters_by_img, native_coco_preds)
    print(f"  [Unmatched75 Breakdown {tag}] {json.dumps(breakdown)}", flush=True)

    return dict(
        checkpoint=str(ckpt_path),
        checkpoint_sha256=reference.file_sha(ckpt_path),
        validator_baseline=off_m,
        unmatched75_breakdown=breakdown,
        methods=method_results,
    )


def _resolve_stage_path(raw: str) -> Path:
    p = Path(raw)
    return p if p.is_absolute() else STAGE / p


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_ckpt",
        default="runs/HistCLS_SchemeA_P2/weights/best.pt",
        help="Base checkpoint path (relative to STAGE or absolute).",
    )
    parser.add_argument(
        "--base_tag",
        default="CLS_P2A_best_E90",
        help="Tag name in output JSON for the base checkpoint.",
    )
    parser.add_argument(
        "--base_metrics_json",
        default="records/HistCLS_SchemeA_P2/evaluation/best/metrics.json",
        help="Optional official metrics.json path for base checkpoint parity gate.",
    )
    parser.add_argument(
        "--extra_ckpt",
        default="",
        help="Optional extra checkpoint (e.g. runs/HistCLS_SchemeA_P2/weights/ema_E60_85_bn_recal.pt)",
    )
    parser.add_argument(
        "--out_json",
        default="diagnostics/step1_box_fusion_replay_s42.json",
        help="Output JSON path (relative to STAGE or absolute).",
    )
    args = parser.parse_args()

    out = dict(
        eval_protocol="SVRDD7 Val 1,000 imgs / 2,612 GTs, FP32 O2M + strict score-preserving intra-cluster replay",
        seed=42,
        checkpoints={},
    )

    base_ck = _resolve_stage_path(args.base_ckpt)
    base_mj = _resolve_stage_path(args.base_metrics_json) if args.base_metrics_json else None
    out["checkpoints"][args.base_tag] = replay_checkpoint(base_ck, args.base_tag, base_mj)

    cand_ema_bn = (
        _resolve_stage_path(args.extra_ckpt)
        if args.extra_ckpt
        else STAGE / "runs/HistCLS_SchemeA_P2/weights/ema_E60_85_bn_recal.pt"
    )
    if cand_ema_bn.is_file():
        out["checkpoints"][cand_ema_bn.stem] = replay_checkpoint(cand_ema_bn, cand_ema_bn.stem, None)

    out_path = _resolve_stage_path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"[DONE] Saved {out_path}", flush=True)


if __name__ == "__main__":
    main()
