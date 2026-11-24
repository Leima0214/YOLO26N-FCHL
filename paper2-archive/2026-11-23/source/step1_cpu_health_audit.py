"""Step 1-3 Pure-CPU Data & Evaluation Health Audit (Train & Val only, NEVER Test).

Performs:
1. Synthetic-area exact custom-slice COCO AP evaluator (parity < 1e-6 vs official small/medium/large),
   evaluating Seed 42 Original CLS (E79) and CLS-P2A (best E90 / best_ap50 E72 / last E100) across
   short640<8, [8,16), >=16, LC+TC<16, and aspect>=4 slices.
2. 1-2 px boundary jitter sensitivity (vertical-only and 4-side random shifts) + vertical tolerance
   diagnostic on unmatched-at-0.75 narrow targets.
3. Train (6,000) vs Val (1,000) spatial, temporal, MD5, dHash, and 16x16 RGB cosine leakage audit
   + side-by-side visual collage (diagnostics/train_val_nearest_pairs.png).
4. Stratified P & MC unmatched prediction audit + 12 unbiased zero-P/MC road halves
   (diagnostics/p_mc_stratified_audit.png).
"""
import contextlib
import copy
import datetime as dt
import hashlib
import io
import json
import os
import re
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

STAGE = Path("/root/YOLO26N-FCHL/reports/paper2_cls_candidate_diagnosis_640_20261120_42106")
REF = Path("/root/YOLO26N-FCHL/reports/paper2_cls_p2p3_snake_640_20261115_42264")
ANN_VAL = Path("/SVRDD/SVRDD_YOLO_READY/annotations/instances_val.json")
TRAIN_TXT = REF / "data" / "train.txt"
VAL_TXT = REF / "data" / "val.txt"
DIAG_DIR = STAGE / "diagnostics"
NAMES = ("LC", "TC", "AC", "P", "MC", "LP", "TP")
GAIN_640 = 640.0 / 1024.0  # 0.625 for SVRDD7 1024x1024 images

FNAME_RE = re.compile(r"^(\d{10})(\d{12})(.+)_(\d+)_(\d+)$")


# ---------------------------------------------------------------------------
# 1. Exact Custom-Slice COCO Evaluator
# ---------------------------------------------------------------------------

def build_coco_dt(gt: COCO, predictions: list[dict]) -> COCO:
    if predictions:
        with contextlib.redirect_stdout(io.StringIO()):
            dt_coco = gt.loadRes(predictions)
    else:
        dt_coco = COCO()
        dt_coco.dataset = {**gt.dataset, "annotations": []}
        dt_coco.createIndex()
    return dt_coco


def official_eval(gt: COCO, dt: COCO, cat_ids: list[int]) -> COCOeval:
    ev = COCOeval(gt, dt, "bbox")
    ev.params.imgIds = sorted(gt.imgs)
    ev.params.catIds = cat_ids
    with contextlib.redirect_stdout(io.StringIO()):
        ev.evaluate()
        ev.accumulate()
    return ev


def mean_valid(arr: np.ndarray) -> float | None:
    v = arr[arr > -1]
    return float(v.mean()) if v.size else None


def extract_coco_metrics(ev: COCOeval, area_idx: int = 0) -> dict:
    p = ev.eval["precision"]  # [T, R, K, A, M]
    r = ev.eval["recall"]     # [T, K, A, M]
    i50 = int(np.argmin(np.abs(ev.params.iouThrs - 0.50)))
    i75 = int(np.argmin(np.abs(ev.params.iouThrs - 0.75)))
    return {
        "AP50_95": mean_valid(p[:, :, :, area_idx, -1]),
        "AP50": mean_valid(p[i50 : i50 + 1, :, :, area_idx, -1]),
        "AP75": mean_valid(p[i75 : i75 + 1, :, :, area_idx, -1]),
        "AR100": mean_valid(r[:, :, area_idx, -1]),
    }


def eval_custom_slice(
    gt_coco: COCO,
    dt_coco: COCO,
    gt_pred_fn,
    dt_pred_fn,
    cat_ids: list[int],
) -> tuple[dict, int, int]:
    """Evaluate exact COCO AP/AR on an arbitrary bbox slice using synthetic areaRng=[[0, 100]].

    Mutates synthetic `area` attributes temporarily and restores original values afterward so
    GT and DT COCO objects can be reused safely across slices without repeated loadRes overhead.
    """
    orig_gt_areas = {aid: ann.get("area") for aid, ann in gt_coco.anns.items()}
    orig_dt_areas = {aid: ann.get("area") for aid, ann in dt_coco.anns.items()}
    cat_set = set(cat_ids)
    try:
        in_gt = 0
        in_imgs = set()
        for ann in gt_coco.anns.values():
            w, h = ann["bbox"][2], ann["bbox"][3]
            keep = (not ann.get("iscrowd", 0)) and (not ann.get("ignore", 0)) and (ann["category_id"] in cat_set)
            if keep and gt_pred_fn(w, h, ann):
                ann["area"] = 1.0
                in_gt += 1
                in_imgs.add(ann["image_id"])
            else:
                ann["area"] = 1e9

        for ann in dt_coco.anns.values():
            w, h = ann["bbox"][2], ann["bbox"][3]
            if (ann["category_id"] in cat_set) and dt_pred_fn(w, h, ann):
                ann["area"] = 1.0
            else:
                ann["area"] = 1e9

        ev = COCOeval(gt_coco, dt_coco, "bbox")
        ev.params.imgIds = sorted(gt_coco.imgs)
        ev.params.catIds = cat_ids
        ev.params.areaRng = [[0.0, 100.0]]
        ev.params.areaRngLbl = ["slice"]
        with contextlib.redirect_stdout(io.StringIO()):
            ev.evaluate()
            ev.accumulate()
        m = extract_coco_metrics(ev, area_idx=0)
        return m, in_gt, len(in_imgs)
    finally:
        for aid, val in orig_gt_areas.items():
            if val is None:
                gt_coco.anns[aid].pop("area", None)
            else:
                gt_coco.anns[aid]["area"] = val
        for aid, val in orig_dt_areas.items():
            if val is None:
                dt_coco.anns[aid].pop("area", None)
            else:
                dt_coco.anns[aid]["area"] = val


def verify_slice_parity(gt: COCO, pred_path: Path, metrics_path: Path, cat_ids: list[int]) -> dict:
    preds = json.loads(pred_path.read_text())
    dt_coco = build_coco_dt(gt, preds)
    ev_off = official_eval(gt, dt_coco, cat_ids)
    official_json = json.loads(metrics_path.read_text())["main"]

    off_all = extract_coco_metrics(ev_off, 0)
    off_s = extract_coco_metrics(ev_off, 1)
    off_m = extract_coco_metrics(ev_off, 2)
    off_l = extract_coco_metrics(ev_off, 3)

    all_m, n_all, _ = eval_custom_slice(gt, dt_coco, lambda w, h, a: True, lambda w, h, a: True, cat_ids)
    s_m, n_s, _ = eval_custom_slice(
        gt, dt_coco,
        lambda w, h, a: 0.0 < a.get("area", w * h) < 32.0 ** 2,
        lambda w, h, a: 0.0 < (w * h) < 32.0 ** 2,
        cat_ids,
    )
    m_m, n_m, _ = eval_custom_slice(
        gt, dt_coco,
        lambda w, h, a: 32.0 ** 2 <= a.get("area", w * h) < 96.0 ** 2,
        lambda w, h, a: 32.0 ** 2 <= (w * h) < 96.0 ** 2,
        cat_ids,
    )
    l_m, n_l, _ = eval_custom_slice(
        gt, dt_coco,
        lambda w, h, a: 96.0 ** 2 <= a.get("area", w * h) < 1e10,
        lambda w, h, a: 96.0 ** 2 <= (w * h) < 1e10,
        cat_ids,
    )

    diffs = {
        "vs_metrics_json_all": max(abs(off_all[k] - official_json[k]) for k in ("AP50", "AP75", "AP50_95", "AR100")),
        "vs_official_all": max(abs(all_m[k] - off_all[k]) for k in ("AP50", "AP75", "AP50_95", "AR100")),
        "vs_official_small": max(abs(s_m[k] - off_s[k]) for k in ("AP50", "AP75", "AP50_95", "AR100")),
        "vs_official_medium": max(abs(m_m[k] - off_m[k]) for k in ("AP50", "AP75", "AP50_95", "AR100")),
        "vs_official_large": max(abs(l_m[k] - off_l[k]) for k in ("AP50", "AP75", "AP50_95", "AR100")),
    }
    max_diff = max(diffs.values())
    assert max_diff < 1e-6, f"Parity check failed: {diffs}"
    return dict(
        max_abs_diff=max_diff,
        diffs=diffs,
        counts=dict(all=n_all, small=n_s, medium=n_m, large=n_l),
        official_pct={k: round(v * 100, 4) for k, v in off_all.items()},
    )


def slice_specs(cats_by_name: dict[str, int]):
    all_cats = [cats_by_name[n] for n in NAMES]
    lc_tc = [cats_by_name["LC"], cats_by_name["TC"]]
    return [
        ("all", lambda w, h, a: True, all_cats),
        ("short640_lt8", lambda w, h, a: min(w, h) * GAIN_640 < 8.0, all_cats),
        ("short640_8_16", lambda w, h, a: 8.0 <= min(w, h) * GAIN_640 < 16.0, all_cats),
        ("short640_ge16", lambda w, h, a: min(w, h) * GAIN_640 >= 16.0, all_cats),
        ("LC_TC_short640_lt16", lambda w, h, a: min(w, h) * GAIN_640 < 16.0, lc_tc),
        ("aspect_ge4", lambda w, h, a: (max(w, h) / max(min(w, h), 1e-6)) >= 4.0, all_cats),
    ]


def evaluate_all_slices(gt: COCO, dt_coco: COCO, cats_by_name: dict[str, int]) -> dict:
    out = {}
    for sname, pred_fn, cids in slice_specs(cats_by_name):
        m, n_gt, n_imgs = eval_custom_slice(gt, dt_coco, pred_fn, pred_fn, cids)
        out[sname] = {
            "n_gt": n_gt,
            "n_gt_images": n_imgs,
            **{k: (round(v * 100.0, 3) if v is not None else None) for k, v in m.items()},
        }
    return out


# ---------------------------------------------------------------------------
# 2. Boundary Jitter Sensitivity & Vertical Tolerance Diagnostic
# ---------------------------------------------------------------------------

def jitter_gt(base_gt: COCO, rng: np.random.Generator, mag640: float, mode: str) -> COCO:
    mag = mag640 / GAIN_640
    gt_j = copy.deepcopy(base_gt)
    for ann in gt_j.anns.values():
        im = gt_j.imgs[ann["image_id"]]
        W, H = float(im["width"]), float(im["height"])
        x, y, w, h = ann["bbox"]
        x1, y1, x2, y2 = x, y, x + w, y + h
        if mode == "vert":
            dy1, dy2 = rng.uniform(-mag, mag, size=2)
            y1_n = float(np.clip(y1 + dy1, 0.0, H - 1.0))
            y2_n = float(np.clip(y2 + dy2, y1_n + 1.0, H))
            ann["bbox"] = [x1, y1_n, x2 - x1, y2_n - y1_n]
        elif mode == "all4":
            dx1, dy1, dx2, dy2 = rng.uniform(-mag, mag, size=4)
            x1_n = float(np.clip(x1 + dx1, 0.0, W - 1.0))
            y1_n = float(np.clip(y1 + dy1, 0.0, H - 1.0))
            x2_n = float(np.clip(x2 + dx2, x1_n + 1.0, W))
            y2_n = float(np.clip(y2 + dy2, y1_n + 1.0, H))
            ann["bbox"] = [x1_n, y1_n, x2_n - x1_n, y2_n - y1_n]
        else:
            raise ValueError(mode)
        ann["area"] = ann["bbox"][2] * ann["bbox"][3]
    return gt_j


def eval_slices_with_fixed_membership(
    orig_gt: COCO,
    jittered_gt: COCO,
    dt_coco: COCO,
    cats_by_name: dict[str, int],
) -> dict:
    """Keep GT slice membership anchored to original (pre-jitter) GT dimensions."""
    out = {}
    orig_anns = orig_gt.anns
    for sname, pred_fn, cids in slice_specs(cats_by_name):
        if sname == "aspect_ge4":
            continue
        m, _, _ = eval_custom_slice(
            jittered_gt,
            dt_coco,
            gt_pred_fn=lambda w, h, a, _fn=pred_fn: _fn(
                orig_anns[a["id"]]["bbox"][2],
                orig_anns[a["id"]]["bbox"][3],
                orig_anns[a["id"]],
            ),
            dt_pred_fn=pred_fn,
            cat_ids=cids,
        )
        out[sname] = {k: round(v * 100.0, 3) for k, v in m.items()}
    return out


def box_iou_1_to_n(box: np.ndarray, arr: np.ndarray) -> np.ndarray:
    lo = np.maximum(arr[:, :2], box[:2])
    hi = np.minimum(arr[:, 2:], box[2:])
    wh = np.maximum(hi - lo, 0.0)
    inter = wh[:, 0] * wh[:, 1]
    a1 = max((box[2] - box[0]) * (box[3] - box[1]), 0.0)
    a2 = np.maximum((arr[:, 2] - arr[:, 0]) * (arr[:, 3] - arr[:, 1]), 0.0)
    return inter / np.maximum(a1 + a2 - inter, 1e-12)


def run_boundary_jitter_and_tolerance(gt: COCO, preds: list[dict], dt_coco: COCO, cats_by_name: dict[str, int]) -> dict:
    base_slices = evaluate_all_slices(gt, dt_coco, cats_by_name)
    configs = [
        ("vert_1px640", 1.0, "vert"),
        ("all4_1px640", 1.0, "all4"),
        ("vert_2px640", 2.0, "vert"),
        ("all4_2px640", 2.0, "all4"),
    ]
    jitter_res = {}
    for cname, mag640, mode in configs:
        rng = np.random.default_rng(42)
        draws = []
        for _ in range(5):
            gt_j = jitter_gt(gt, rng, mag640, mode)
            draws.append(eval_slices_with_fixed_membership(gt, gt_j, dt_coco, cats_by_name))
        summary = {}
        for sname in ("all", "short640_lt8", "short640_8_16", "short640_ge16", "LC_TC_short640_lt16"):
            summary[sname] = {}
            for k in ("AP50_95", "AP50", "AP75", "AR100"):
                vals = [d[sname][k] for d in draws]
                mean_v = float(np.mean(vals))
                summary[sname][k] = {
                    "mean": round(mean_v, 3),
                    "std": round(float(np.std(vals)), 3),
                    "delta_vs_clean": round(mean_v - base_slices[sname][k], 3),
                }
        jitter_res[cname] = summary

    # Per-GT diagnostic vertical tolerance (same-image, same-class predictions)
    by_img_cls = defaultdict(list)
    for p in preds:
        x, y, w, h = p["bbox"]
        by_img_cls[(p["image_id"], p["category_id"])].append([x, y, x + w, y + h, p["score"]])
    by_img_cls = {k: np.array(v, dtype=np.float32) for k, v in by_img_cls.items()}

    cat_names = {v: k for k, v in cats_by_name.items()}
    tol_buckets = defaultdict(list)
    sh1 = np.array([-1.0, 0.0, 1.0], dtype=np.float32) / GAIN_640
    sh2_extra = np.array([-2.0, 2.0], dtype=np.float32) / GAIN_640

    for ann in gt.anns.values():
        if ann.get("iscrowd") or ann.get("ignore"):
            continue
        x, y, w, h = ann["bbox"]
        short640 = min(w, h) * GAIN_640
        cn = cat_names[ann["category_id"]]
        tb = np.array([x, y, x + w, y + h], dtype=np.float32)
        arr = by_img_cls.get((ann["image_id"], ann["category_id"]))
        if arr is None or len(arr) == 0:
            rec = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        else:
            ious0 = box_iou_1_to_n(tb, arr[:, :4])
            best0 = float(ious0.max())
            best1 = best0
            for d in sh1:
                if d == 0.0:
                    continue
                a_s = arr[:, :4].copy()
                a_s[:, 1] += d; a_s[:, 3] += d
                best1 = max(best1, float(box_iou_1_to_n(tb, a_s).max()))
            best2 = best1
            for d in sh2_extra:
                a_s = arr[:, :4].copy()
                a_s[:, 1] += d; a_s[:, 3] += d
                best2 = max(best2, float(box_iou_1_to_n(tb, a_s).max()))

            m25 = arr[:, 4] >= 0.25
            if not np.any(m25):
                b0_25 = b1_25 = b2_25 = 0.0
            else:
                a25 = arr[m25, :4]
                b0_25 = float(box_iou_1_to_n(tb, a25).max())
                b1_25 = b0_25
                for d in sh1:
                    if d == 0.0:
                        continue
                    as25 = a25.copy(); as25[:, 1] += d; as25[:, 3] += d
                    b1_25 = max(b1_25, float(box_iou_1_to_n(tb, as25).max()))
                b2_25 = b1_25
                for d in sh2_extra:
                    as25 = a25.copy(); as25[:, 1] += d; as25[:, 3] += d
                    b2_25 = max(b2_25, float(box_iou_1_to_n(tb, as25).max()))
            rec = (best0, best1, best2, b0_25, b1_25, b2_25)

        tol_buckets["all"].append(rec)
        if short640 < 8.0:
            tol_buckets["short640_lt8"].append(rec)
        elif short640 < 16.0:
            tol_buckets["short640_8_16"].append(rec)
        else:
            tol_buckets["short640_ge16"].append(rec)
        if cn in ("LC", "TC") and short640 < 16.0:
            tol_buckets["LC_TC_short640_lt16"].append(rec)

    tol_summary = {}
    for k, recs in tol_buckets.items():
        arr_r = np.array(recs, dtype=np.float64)
        n = len(arr_r)
        clean75 = arr_r[:, 0] >= 0.75
        tol1 = arr_r[:, 1] >= 0.75
        tol2 = arr_r[:, 2] >= 0.75
        c25_0 = arr_r[:, 3] >= 0.75
        c25_1 = arr_r[:, 4] >= 0.75
        c25_2 = arr_r[:, 5] >= 0.75
        med_unmatched = (arr_r[:, 0] >= 0.50) & (arr_r[:, 0] < 0.75)
        med_to_1 = med_unmatched & tol1
        med_to_2 = med_unmatched & tol2
        tol_summary[k] = dict(
            n_gt=n,
            raw_gt075_all_scores=int(clean75.sum()),
            pct_gt075_all_scores=round(100.0 * clean75.mean(), 2),
            tol_1px_gt075_all_scores=int(tol1.sum()),
            gain_1px_count=int((~clean75 & tol1).sum()),
            tol_2px_gt075_all_scores=int(tol2.sum()),
            gain_2px_count=int((~clean75 & tol2).sum()),
            med_iou_050_075_count=int(med_unmatched.sum()),
            med_recovered_by_1px_vert=int(med_to_1.sum()),
            med_recovered_by_1px_pct=round(100.0 * med_to_1.sum() / max(1, int(med_unmatched.sum())), 2),
            med_recovered_by_2px_vert=int(med_to_2.sum()),
            med_recovered_by_2px_pct=round(100.0 * med_to_2.sum() / max(1, int(med_unmatched.sum())), 2),
            at025_clean75=int(c25_0.sum()),
            at025_tol1_gain=int((~c25_0 & c25_1).sum()),
            at025_tol2_gain=int((~c25_0 & c25_2).sum()),
        )

    return dict(
        clean_slices=base_slices,
        gt_random_jitter_5draws=jitter_res,
        vertical_tolerance_diagnostic=tol_summary,
    )


# ---------------------------------------------------------------------------
# 3. Train / Val Spatial, Temporal & Visual Leakage Audit
# ---------------------------------------------------------------------------

def _img_sig_worker(path_str: str):
    p = Path(path_str)
    data = p.read_bytes()
    md5 = hashlib.md5(data).hexdigest()
    arr = np.frombuffer(data, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    thumb16 = cv2.resize(rgb, (16, 16), interpolation=cv2.INTER_AREA).astype(np.float32).reshape(-1)
    norm = float(np.linalg.norm(thumb16))
    if norm > 1e-9:
        thumb16 /= norm
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    diff = small[:, 1:] > small[:, :-1]
    bits = diff.reshape(-1).astype(np.uint64)
    dhash = np.uint64(0)
    for b in bits:
        dhash = (dhash << np.uint64(1)) | b
    return path_str, md5, thumb16, int(dhash)


def parse_svrdd_meta(path_str: str) -> dict:
    stem = Path(path_str).stem
    m = FNAME_RE.match(stem)
    if not m:
        return dict(parsed=False, stem=stem)
    pref10, ymdhms, cam, x_str, y_str = m.groups()
    ts = dt.datetime.strptime("20" + ymdhms, "%Y%m%d%H%M%S")
    return dict(
        parsed=True,
        stem=stem,
        prefix10=pref10,
        date=ymdhms[:6],
        cam=cam,
        ts_epoch=ts.timestamp(),
        x=int(x_str),
        y=int(y_str),
    )


def quantiles_dict(arr: np.ndarray, qs=(0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100)):
    return {f"p{q}": round(float(np.percentile(arr, q)), 4) for q in qs}


def run_train_val_leakage_audit() -> dict:
    train_paths = [x.strip() for x in TRAIN_TXT.read_text().splitlines() if x.strip()]
    val_paths = [x.strip() for x in VAL_TXT.read_text().splitlines() if x.strip()]
    assert len(train_paths) == 6000 and len(val_paths) == 1000
    assert set(Path(x).name for x in train_paths).isdisjoint(Path(x).name for x in val_paths)

    all_paths = train_paths + val_paths
    md5s, thumbs, dhashes = [], [], []
    with ProcessPoolExecutor(max_workers=4) as ex:
        for _, md5, th, dh in ex.map(_img_sig_worker, all_paths, chunksize=100):
            md5s.append(md5)
            thumbs.append(th)
            dhashes.append(dh)

    tr_md5, va_md5 = md5s[:6000], md5s[6000:]
    tr_th = np.stack(thumbs[:6000], axis=0)
    va_th = np.stack(thumbs[6000:], axis=0)
    tr_dh = np.array(dhashes[:6000], dtype=np.uint64)
    va_dh = np.array(dhashes[6000:], dtype=np.uint64)

    exact_md5_dups = set(va_md5) & set(tr_md5)

    cos_mat = va_th @ tr_th.T
    top1_cos_idx = np.argmax(cos_mat, axis=1)
    top1_cos = cos_mat[np.arange(1000), top1_cos_idx]

    lut = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
    xor_u64 = np.bitwise_xor(va_dh[:, None], tr_dh[None, :])
    xor_bytes = xor_u64.view(np.uint8).reshape(1000, 6000, 8)
    ham_mat = lut[xor_bytes].sum(axis=2, dtype=np.uint8)
    min_ham_idx = np.argmin(ham_mat, axis=1)
    min_ham = ham_mat[np.arange(1000), min_ham_idx]

    tr_meta = [parse_svrdd_meta(p) for p in train_paths]
    va_meta = [parse_svrdd_meta(p) for p in val_paths]
    assert all(m["parsed"] for m in tr_meta) and all(m["parsed"] for m in va_meta)

    tr_xy = np.array([[m["x"], m["y"]] for m in tr_meta], dtype=np.float64)
    va_xy = np.array([[m["x"], m["y"]] for m in va_meta], dtype=np.float64)
    tr_x_span = (float(tr_xy[:, 0].min()), float(tr_xy[:, 0].max()))
    tr_y_span = (float(tr_xy[:, 1].min()), float(tr_xy[:, 1].max()))

    dx = va_xy[:, 0:1] - tr_xy[:, 0:1].T
    dy = va_xy[:, 1:2] - tr_xy[:, 1:2].T
    dist_raw = np.hypot(dx, dy)
    nn_dist_idx = np.argmin(dist_raw, axis=1)
    nn_dist_raw = dist_raw[np.arange(1000), nn_dist_idx]
    nn_dist_m = nn_dist_raw / 100.0

    tr_by_date = defaultdict(list)
    for i, m in enumerate(tr_meta):
        tr_by_date[m["date"]].append((i, m["ts_epoch"]))
    min_dt_s = np.full(1000, np.inf, dtype=np.float64)
    for vi, vm in enumerate(va_meta):
        cands = tr_by_date.get(vm["date"])
        if cands:
            _, best_dt = min(((ti, abs(vm["ts_epoch"] - te)) for ti, te in cands), key=lambda x: x[1])
            min_dt_s[vi] = best_dt

    tr_pref_date = set((m["prefix10"], m["date"]) for m in tr_meta)
    va_pref_date = set((m["prefix10"], m["date"]) for m in va_meta)

    vis_order = np.argsort(-top1_cos)
    top6_vis = [(int(vi), int(top1_cos_idx[vi])) for vi in vis_order[:6]]
    used_vi = {x[0] for x in top6_vis}
    sp_order = np.argsort(nn_dist_m)
    top6_sp = []
    for vi in sp_order:
        vi = int(vi)
        if vi not in used_vi:
            top6_sp.append((vi, int(nn_dist_idx[vi])))
            used_vi.add(vi)
        if len(top6_sp) == 6:
            break

    font = ImageFont.load_default()
    cell_w, cell_h = 260, 260
    hdr_h = 40
    gap = 6
    cols = 2
    pair_w = cell_w * 2 + gap
    pair_h = cell_h + hdr_h
    margin = 16
    sec_h = 28
    cw = margin * 2 + cols * pair_w + (cols - 1) * 16
    ch = margin * 2 + sec_h * 2 + 6 * (pair_h + 10) + 16
    canvas = Image.new("RGB", (cw, ch), (245, 247, 250))
    draw = ImageDraw.Draw(canvas)

    def draw_pair(vi, ti, px, py):
        vp = Path(val_paths[vi])
        tp = Path(train_paths[ti])
        cos_v = float(cos_mat[vi, ti])
        ham_v = int(ham_mat[vi, ti])
        dm = float(dist_raw[vi, ti] / 100.0)
        same_d = va_meta[vi]["date"] == tr_meta[ti]["date"]
        dts = abs(va_meta[vi]["ts_epoch"] - tr_meta[ti]["ts_epoch"]) if same_d else -1.0
        dt_lbl = f"{dts:.0f}s" if same_d else f"diffDate({va_meta[vi]['date']} vs {tr_meta[ti]['date']})"
        draw.text((px, py), f"V:{vp.stem[:22]}.. vs T:{tp.stem[:22]}..", fill=(20, 20, 20), font=font)
        draw.text((px, py + 16), f"cos={cos_v:.4f} | dHash={ham_v} | dist={dm:.2f}m | dt={dt_lbl}", fill=(170, 25, 25), font=font)
        vim = Image.open(vp).convert("RGB").resize((cell_w, cell_h), Image.Resampling.BILINEAR)
        tim = Image.open(tp).convert("RGB").resize((cell_w, cell_h), Image.Resampling.BILINEAR)
        canvas.paste(vim, (px, py + hdr_h))
        canvas.paste(tim, (px + cell_w + gap, py + hdr_h))
        draw.text((px + 6, py + hdr_h + 6), "VAL", fill=(255, 220, 0), font=font)
        draw.text((px + cell_w + gap + 6, py + hdr_h + 6), "TRAIN", fill=(0, 255, 180), font=font)

    y_cur = margin
    draw.text((margin, y_cur), "TOP-6 VAL-TRAIN BY VISUAL 16x16 RGB COSINE SIMILARITY", fill=(10, 40, 120), font=font)
    y_cur += sec_h
    for idx, (vi, ti) in enumerate(top6_vis):
        r, c = divmod(idx, cols)
        draw_pair(vi, ti, margin + c * (pair_w + 16), y_cur + r * (pair_h + 10))
    y_cur += 3 * (pair_h + 10) + 12
    draw.text((margin, y_cur), "TOP-6 CLOSEST SPATIAL DISTANCE (EXCL. TOP-6 VISUAL)", fill=(120, 30, 10), font=font)
    y_cur += sec_h
    for idx, (vi, ti) in enumerate(top6_sp):
        r, c = divmod(idx, cols)
        draw_pair(vi, ti, margin + c * (pair_w + 16), y_cur + r * (pair_h + 10))

    collage_path = DIAG_DIR / "train_val_nearest_pairs.png"
    canvas.save(collage_path)

    valid_dt = min_dt_s[np.isfinite(min_dt_s)]
    return dict(
        train_count=len(train_paths),
        val_count=len(val_paths),
        exact_md5_duplicates=len(exact_md5_dups),
        filename_parse_ok=dict(train=6000, val=1000),
        coordinate_convention="Baidu Mercator * 100 (cm): raw/100.0 = meters; verified Beijing span ~[12.90M..13.00M, 4.81M..4.88M] m",
        train_raw_coord_span=dict(x=tr_x_span, y=tr_y_span),
        same_prefix10_date_overlap=dict(val_groups=len(va_pref_date), train_groups=len(tr_pref_date), overlap=len(va_pref_date & tr_pref_date)),
        visual_cosine_top1=dict(
            quantiles=quantiles_dict(top1_cos),
            ge_099=int((top1_cos >= 0.99).sum()),
            ge_098=int((top1_cos >= 0.98).sum()),
            ge_095=int((top1_cos >= 0.95).sum()),
            ge_090=int((top1_cos >= 0.90).sum()),
            max=round(float(top1_cos.max()), 5),
        ),
        dhash_hamming_min=dict(
            quantiles=quantiles_dict(min_ham),
            eq_0=int((min_ham == 0).sum()),
            le_4=int((min_ham <= 4).sum()),
            le_8=int((min_ham <= 8).sum()),
            le_12=int((min_ham <= 12).sum()),
            min=int(min_ham.min()),
        ),
        spatial_nn_distance_m=dict(
            quantiles=quantiles_dict(nn_dist_m),
            eq_0m=int((nn_dist_m == 0.0).sum()),
            le_2m=int((nn_dist_m <= 2.0).sum()),
            le_5m=int((nn_dist_m <= 5.0).sum()),
            le_10m=int((nn_dist_m <= 10.0).sum()),
            le_25m=int((nn_dist_m <= 25.0).sum()),
            le_50m=int((nn_dist_m <= 50.0).sum()),
            le_100m=int((nn_dist_m <= 100.0).sum()),
            min_m=round(float(nn_dist_m.min()), 2),
        ),
        same_date_temporal_nn_s=dict(
            val_with_same_date_train=int(len(valid_dt)),
            quantiles=quantiles_dict(valid_dt) if len(valid_dt) else {},
            le_2s=int((valid_dt <= 2.0).sum()),
            le_5s=int((valid_dt <= 5.0).sum()),
            le_10s=int((valid_dt <= 10.0).sum()),
            le_30s=int((valid_dt <= 30.0).sum()),
            le_120s=int((valid_dt <= 120.0).sum()),
            le_600s=int((valid_dt <= 600.0).sum()),
        ),
        collage_path=str(collage_path),
    )


# ---------------------------------------------------------------------------
# 4. Stratified P & MC Unmatched Prediction Audit + Unbiased Zero-P/MC Audit
# ---------------------------------------------------------------------------

def run_p_mc_visual_audit(gt: COCO, preds: list[dict], cats_by_name: dict[str, int]) -> dict:
    val_map = {Path(x).name: Path(x) for x in VAL_TXT.read_text().splitlines() if x.strip()}
    cat_id_to_name = {v: k for k, v in cats_by_name.items()}
    gt_by_img = defaultdict(list)
    for ann in gt.anns.values():
        if ann.get("iscrowd") or ann.get("ignore"):
            continue
        x, y, w, h = ann["bbox"]
        gt_by_img[ann["image_id"]].append((cat_id_to_name[ann["category_id"]], np.array([x, y, x + w, y + h], dtype=np.float32)))

    strata = [("[0.25,0.45)", 0.25, 0.45), ("[0.45,0.65)", 0.45, 0.65), (">=0.65", 0.65, 2.0)]
    counts = {cn: {s[0]: {"iou_any_lt01": 0, "iou_same_lt01": 0} for s in strata} for cn in ("P", "MC")}
    pools = {cn: {s[0]: [] for s in strata} for cn in ("P", "MC")}

    for p in preds:
        cn = cat_id_to_name[p["category_id"]]
        if cn not in ("P", "MC") or p["score"] < 0.25:
            continue
        x, y, w, h = p["bbox"]
        pb = np.array([x, y, x + w, y + h], dtype=np.float32)
        gts = gt_by_img.get(p["image_id"], [])
        if gts:
            arr_all = np.stack([g[1] for g in gts], axis=0)
            max_any = float(box_iou_1_to_n(pb, arr_all).max())
            same_list = [g[1] for g in gts if g[0] == cn]
            max_same = float(box_iou_1_to_n(pb, np.stack(same_list, axis=0)).max()) if same_list else 0.0
        else:
            max_any = max_same = 0.0

        for sname, lo, hi in strata:
            if lo <= p["score"] < hi:
                if max_same < 0.10:
                    counts[cn][sname]["iou_same_lt01"] += 1
                if max_any < 0.10:
                    counts[cn][sname]["iou_any_lt01"] += 1
                    pools[cn][sname].append(dict(
                        image_id=p["image_id"],
                        fname=Path(gt.imgs[p["image_id"]]["file_name"]).name,
                        cls=cn,
                        score=float(p["score"]),
                        bbox=list(p["bbox"]),
                    ))
                break

    rng = np.random.default_rng(42)
    selected_crops = []
    for sname, _, _ in strata:
        for cn in ("P", "MC"):
            pool = sorted(pools[cn][sname], key=lambda r: (-round(r["score"], 6), r["image_id"], r["bbox"]))
            idxs = rng.choice(len(pool), size=min(3, len(pool)), replace=False) if pool else []
            for ix in idxs:
                selected_crops.append(dict(pool[int(ix)], stratum=sname))

    zero_p_mc_ids = sorted(
        img_id for img_id in gt.imgs
        if not any(g[0] in ("P", "MC") for g in gt_by_img.get(img_id, []))
    )
    unbiased_ids = [int(zero_p_mc_ids[int(ix)]) for ix in rng.choice(len(zero_p_mc_ids), size=12, replace=False)]

    font = ImageFont.load_default()
    crop_sz = 200
    hdr = 30
    cols_c = 6
    road_w, road_h = 412, 214
    margin = 16
    cw = max(margin * 2 + cols_c * (crop_sz + 8), margin * 2 + 3 * (road_w + 12))
    rows_c = (len(selected_crops) + cols_c - 1) // cols_c
    rows_u = (len(unbiased_ids) + 2) // 3
    ch = margin * 2 + 28 + rows_c * (crop_sz + hdr + 8) + 32 + rows_u * (road_h + hdr + 10)
    canvas = Image.new("RGB", (cw, ch), (246, 248, 250))
    draw = ImageDraw.Draw(canvas)

    y_cur = margin
    draw.text((margin, y_cur), "STRATIFIED UNMATCHED (IoU_any < 0.1) P & MC PREDICTIONS (RED=PRED, GREEN=ANY GT)", fill=(20, 20, 20), font=font)
    y_cur += 28
    for idx, item in enumerate(selected_crops):
        r, c = divmod(idx, cols_c)
        px = margin + c * (crop_sz + 8)
        py = y_cur + r * (crop_sz + hdr + 8)
        im_path = val_map[item["fname"]]
        rgb = cv2.cvtColor(cv2.imread(str(im_path)), cv2.COLOR_BGR2RGB)
        H, W = rgb.shape[:2]
        bx, by, bw, bh = item["bbox"]
        win = max(160, int(round(max(bw, bh) * 2.4)))
        cx = int(round(bx + bw * 0.5))
        cy = int(round(by + bh * 0.5))
        x0 = max(0, min(W - win, cx - win // 2))
        y0 = max(0, min(H - win, cy - win // 2))
        x1 = min(W, x0 + win)
        y1 = min(H, y0 + win)
        patch = Image.fromarray(rgb[y0:y1, x0:x1]).resize((crop_sz, crop_sz), Image.Resampling.BILINEAR)
        sx = crop_sz / max(1, x1 - x0)
        sy = crop_sz / max(1, y1 - y0)
        pd = ImageDraw.Draw(patch)
        for gcls, gb in gt_by_img.get(item["image_id"], []):
            if gb[2] > x0 and gb[0] < x1 and gb[3] > y0 and gb[1] < y1:
                pd.rectangle([(gb[0] - x0) * sx, (gb[1] - y0) * sy, (gb[2] - x0) * sx, (gb[3] - y0) * sy], outline=(0, 220, 80), width=2)
        pd.rectangle([(bx - x0) * sx, (by - y0) * sy, (bx + bw - x0) * sx, (by + bh - y0) * sy], outline=(255, 40, 40), width=2)
        draw.text((px, py), f"{item['cls']} {item['stratum']} s={item['score']:.2f}", fill=(170, 20, 20), font=font)
        draw.text((px, py + 13), f"{item['fname'][:24]}", fill=(60, 60, 60), font=font)
        canvas.paste(patch, (px, py + hdr))

    y_cur += rows_c * (crop_sz + hdr + 8) + 10
    draw.text((margin, y_cur), "12 UNBIASED RANDOM VAL ROAD HALVES [48%,100%] WITH ZERO P/MC GT (GREEN=EXISTING NON-P/MC GTs)", fill=(10, 50, 120), font=font)
    y_cur += 24
    for idx, img_id in enumerate(unbiased_ids):
        r, c = divmod(idx, 3)
        px = margin + c * (road_w + 12)
        py = y_cur + r * (road_h + hdr + 10)
        fname = Path(gt.imgs[img_id]["file_name"]).name
        rgb = cv2.cvtColor(cv2.imread(str(val_map[fname])), cv2.COLOR_BGR2RGB)
        H, W = rgb.shape[:2]
        y0 = int(round(0.48 * H))
        patch = Image.fromarray(rgb[y0:H, :]).resize((road_w, road_h), Image.Resampling.BILINEAR)
        sx = road_w / W
        sy = road_h / (H - y0)
        pd = ImageDraw.Draw(patch)
        for gcls, gb in gt_by_img.get(img_id, []):
            if gb[3] > y0:
                pd.rectangle([gb[0] * sx, max(0.0, gb[1] - y0) * sy, gb[2] * sx, (gb[3] - y0) * sy], outline=(0, 220, 80), width=2)
        n_other = len(gt_by_img.get(img_id, []))
        draw.text((px, py), f"id={img_id} otherGT={n_other} {fname[:28]}", fill=(20, 40, 100), font=font)
        canvas.paste(patch, (px, py + hdr))

    out_png = DIAG_DIR / "p_mc_stratified_audit.png"
    canvas.save(out_png)
    return dict(
        unmatched_counts_ge025=counts,
        zero_p_mc_val_images=len(zero_p_mc_ids),
        selected_crops=selected_crops,
        unbiased_road_image_ids=unbiased_ids,
        collage_path=str(out_png),
    )


def main():
    assert os.environ.get("CUDA_VISIBLE_DEVICES", "") == "", "Run with CUDA_VISIBLE_DEVICES=''!"
    gt = COCO(str(ANN_VAL))
    cats_by_name = {c["name"]: c["id"] for c in gt.cats.values()}
    assert set(cats_by_name) == set(NAMES)
    all_cats = [cats_by_name[n] for n in NAMES]

    p2a_best_pred = STAGE / "records/HistCLS_SchemeA_P2/evaluation/best/predictions.coco.json"
    p2a_best_metrics = STAGE / "records/HistCLS_SchemeA_P2/evaluation/best/metrics.json"
    parity = verify_slice_parity(gt, p2a_best_pred, p2a_best_metrics, all_cats)
    print(f"[Parity OK] max_abs_diff={parity['max_abs_diff']:.2e}", flush=True)

    checkpoints = [
        ("HistoricalCLS_best_E79", STAGE / "records/HistoricalCLS_DualBest/evaluation/best/predictions.coco.json"),
        ("CLS_P2A_best_E90", p2a_best_pred),
        ("CLS_P2A_best_ap50_E72", STAGE / "records/HistCLS_SchemeA_P2/evaluation/best_ap50/predictions.coco.json"),
        ("CLS_P2A_last_E100", STAGE / "records/HistCLS_SchemeA_P2/evaluation/last/predictions.coco.json"),
    ]
    slice_evals = {}
    p2a_preds = None
    p2a_dt = None
    for label, ppath in checkpoints:
        preds = json.loads(ppath.read_text())
        dt_c = build_coco_dt(gt, preds)
        slice_evals[label] = evaluate_all_slices(gt, dt_c, cats_by_name)
        s = slice_evals[label]
        print(
            f"  [{label}] all={s['all']['AP50_95']}/{s['all']['AP50']}/{s['all']['AP75']} | "
            f"lt8={s['short640_lt8']['AP50_95']}/{s['short640_lt8']['AP75']} | "
            f"8_16={s['short640_8_16']['AP50_95']}/{s['short640_8_16']['AP75']} | "
            f"aspect>=4={s['aspect_ge4']['AP50_95']}",
            flush=True,
        )
        if label == "CLS_P2A_best_E90":
            p2a_preds = preds
            p2a_dt = dt_c

    jitter = run_boundary_jitter_and_tolerance(gt, p2a_preds, p2a_dt, cats_by_name)
    print("[Boundary jitter + tolerance OK]", flush=True)

    leakage = run_train_val_leakage_audit()
    print(
        f"[Leakage Audit OK] md5_dup={leakage['exact_md5_duplicates']}, "
        f"cos>=0.98={leakage['visual_cosine_top1']['ge_098']}, "
        f"dHash<=4={leakage['dhash_hamming_min']['le_4']}, "
        f"nn<=10m={leakage['spatial_nn_distance_m']['le_10m']}",
        flush=True,
    )

    pmc = run_p_mc_visual_audit(gt, p2a_preds, cats_by_name)
    print("[P/MC Stratified Audit OK]", json.dumps(pmc["unmatched_counts_ge025"]), flush=True)

    out = dict(
        generated_at=dt.datetime.now().isoformat(),
        seed=42,
        parity_check=parity,
        custom_slice_coco_eval=slice_evals,
        boundary_jitter_and_tolerance=jitter,
        train_val_leakage=leakage,
        p_mc_audit=pmc,
    )
    out_json = DIAG_DIR / "step1_cpu_health_audit.json"
    out_json.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"[DONE] Saved {out_json}", flush=True)


if __name__ == "__main__":
    main()
