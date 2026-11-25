import contextlib
import io
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch
import torchvision
from pycocotools.coco import COCO

STAGE = Path("/root/YOLO26N-FCHL/reports/paper2_cls_candidate_diagnosis_640_20261120_42106")
GEO = Path("/root/YOLO26N-FCHL/reports/paper2_geometry_semantic_640_20261109_42266")
REF = Path("/root/YOLO26N-FCHL/reports/paper2_cls_p2p3_snake_640_20261115_42264")
sys.path[:0] = [str(STAGE), str(GEO / "source"), str(REF / "source/dependencies"), str(GEO / "source/dependencies"), "/root/YOLO26N-FCHL"]

import candidates
import models
import experiment as reference
from run import DATA_YAML, dump
from ultralytics.utils import ops


def print_training_stats():
    print("=== H1_L1 GAMMA ===")
    for name in ("best.pt", "last.pt"):
        ck = torch.load(STAGE / "runs/H1_L1/weights" / name, map_location="cpu", weights_only=False)
        m = ck.get("ema") or ck["model"]
        print(f"  H1_L1 {name}: epoch={ck.get('epoch')} gamma={float(m.model[1].gamma.detach().float().item()):.6f}")

    print("\n=== EPOCH CURVES (C0 vs HistoricalCLS vs H1_L1) ===")
    for arm in ("C0", "HistoricalCLS", "H1_L1"):
        ep_file = STAGE / "records" / arm / "epochs.jsonl"
        if not ep_file.exists():
            continue
        rows = [json.loads(x) for x in ep_file.read_text().splitlines()]
        m0 = rows[0]["metrics"]
        k95 = "COCO/AP50_95" if "COCO/AP50_95" in m0 else "metrics/mAP50-95(B)"
        k50 = "COCO/AP50" if "COCO/AP50" in m0 else "metrics/mAP50(B)"
        k75 = "COCO/AP75" if "COCO/AP75" in m0 else "metrics/AP75(B)"
        b95 = max(rows, key=lambda r: r["metrics"][k95])
        b50 = max(rows, key=lambda r: r["metrics"][k50])
        b75 = max(rows, key=lambda r: r["metrics"][k75])
        e80_90 = [r["metrics"][k95] for r in rows if 80 <= r["epoch"] <= 90]
        e91_100 = [r["metrics"][k95] for r in rows if 91 <= r["epoch"] <= 100]
        e80_90_50 = [r["metrics"][k50] for r in rows if 80 <= r["epoch"] <= 90]
        print(f"{arm} [{k95}]:")
        print(f"  Best AP50_95: E{b95['epoch']} -> AP50_95={b95['metrics'][k95]:.5f}, AP50={b95['metrics'][k50]:.5f}, AP75={b95['metrics'][k75]:.5f}")
        print(f"  Best AP50   : E{b50['epoch']} -> AP50={b50['metrics'][k50]:.5f}, AP50_95={b50['metrics'][k95]:.5f}, AP75={b50['metrics'][k75]:.5f}")
        print(f"  Best AP75   : E{b75['epoch']} -> AP75={b75['metrics'][k75]:.5f}, AP50_95={b75['metrics'][k95]:.5f}")
        print(f"  E80-90 mean : AP50_95={sum(e80_90)/len(e80_90):.5f} (AP50={sum(e80_90_50)/len(e80_90_50):.5f}) | E91-100 mean: {sum(e91_100)/len(e91_100):.5f}")
        print(f"  E89->90->91->100: E89={rows[88]['metrics'][k95]:.5f}, E90={rows[89]['metrics'][k95]:.5f}, E91={rows[90]['metrics'][k95]:.5f}, E100={rows[99]['metrics'][k95]:.5f}")


class RawExtractor(reference.TaskValidator):
    def postprocess(self, pred):
        decoded, _ = pred
        assert not self.end2end and decoded.shape[1] == 11
        self.raw_batch = decoded.detach().float().cpu()
        return [dict(bboxes=torch.zeros((0, 4)), conf=torch.zeros((0,)), cls=torch.zeros((0,)), extra=torch.zeros((0, 0))) for _ in range(decoded.shape[0])]

    def update_metrics(self, preds, batch):
        y = self.raw_batch
        boxes_xyxy = ops.xywh2xyxy(y[:, :4].transpose(1, 2))
        scores = y[:, 4:].transpose(1, 2)
        for si, name in enumerate(batch["im_file"]):
            pb = self._prepare_batch(si, batch)
            image_id = self.img_ids[Path(name).name]
            sc = scores[si]
            # Keep only (anchor, cls) entries > 0.001 (multi_label=True, exact YOLO behavior)
            i_idx, c_idx = (sc > 0.001).nonzero(as_tuple=False).T
            if len(i_idx) == 0:
                self.items.append((image_id, np.zeros((0, 4), np.float32), np.zeros((0,), np.float32), np.zeros((0,), np.int64), np.zeros((0, 4), np.float32)))
                continue
            b_net = boxes_xyxy[si, i_idx]
            s_val = sc[i_idx, c_idx]
            # YOLO pre-NMS top-30000 cap
            if len(s_val) > 30000:
                order = s_val.argsort(descending=True)[:30000]
                b_net, s_val, c_idx = b_net[order], s_val[order], c_idx[order]
            else:
                order = s_val.argsort(descending=True)
                b_net, s_val, c_idx = b_net[order], s_val[order], c_idx[order]
            b_ori = ops.scale_boxes(pb["imgsz"], b_net.clone(), pb["ori_shape"], pb["ratio_pad"])
            self.items.append((image_id, b_net.numpy(), s_val.numpy(), c_idx.numpy(), b_ori.numpy()))
        self.seen += len(batch["im_file"])

    def get_stats(self):
        return {}

    def print_results(self):
        pass


def extract_raw(ckpt_path, gt, img_ids):
    val = RawExtractor(
        args=dict(
            task="detect", data=str(DATA_YAML), imgsz=640, batch=32, workers=4,
            device="0", conf=0.001, iou=0.7, max_det=300, rect=True,
            save_json=False, plots=False, verbose=False, half=False, split="val",
        ),
        save_dir=STAGE / "diagnostics/_tmp_probe",
    )
    val.img_ids = img_ids
    val.items = []
    with contextlib.redirect_stdout(io.StringIO()):
        val(model=reference.load_model(ckpt_path))
    assert val.seen == 1000
    return val.items


def box_iou_matrix(a, b):
    # a: (N,4), b: (M,4) numpy
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.maximum(rb - lt, 0.0)
    inter = wh[..., 0] * wh[..., 1]
    area_a = np.maximum(a[:, 2] - a[:, 0], 0.0) * np.maximum(a[:, 3] - a[:, 1], 0.0)
    area_b = np.maximum(b[:, 2] - b[:, 0], 0.0) * np.maximum(b[:, 3] - b[:, 1], 0.0)
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-12)


def run_std_nms(items, iou_thr, max_det=300):
    preds = []
    for image_id, b_net, scores, clses, b_ori in items:
        if len(scores) == 0:
            continue
        bt = torch.from_numpy(b_net)
        st = torch.from_numpy(scores)
        ct = torch.from_numpy(clses)
        offsets = ct.float() * 7680.0
        keep = torchvision.ops.nms(bt + offsets[:, None], st, iou_thr)[:max_det].numpy()
        for k in keep:
            x1, y1, x2, y2 = b_ori[k]
            preds.append((image_id, int(clses[k]), [float(x1), float(y1), float(x2 - x1), float(y2 - y1)], float(scores[k])))
    return preds


def run_diou_nms(items, iou_thr, beta=1.0, max_det=300):
    preds = []
    for image_id, b_net, scores, clses, b_ori in items:
        if len(scores) == 0:
            continue
        kept_all = []
        for c in np.unique(clses):
            m = np.flatnonzero(clses == c)
            bn = b_net[m]
            sc = scores[m]
            order = np.argsort(-sc)
            bn, sc, m = bn[order], sc[order], m[order]
            cx = 0.5 * (bn[:, 0] + bn[:, 2])
            cy = 0.5 * (bn[:, 1] + bn[:, 3])
            areas = np.maximum(bn[:, 2] - bn[:, 0], 0.0) * np.maximum(bn[:, 3] - bn[:, 1], 0.0)
            active = np.ones(len(m), dtype=bool)
            for i in range(len(m)):
                if not active[i]:
                    continue
                kept_all.append(m[i])
                if i + 1 >= len(m):
                    break
                rem = np.flatnonzero(active[i + 1:]) + (i + 1)
                if len(rem) == 0:
                    break
                x1 = np.maximum(bn[i, 0], bn[rem, 0])
                y1 = np.maximum(bn[i, 1], bn[rem, 1])
                x2 = np.minimum(bn[i, 2], bn[rem, 2])
                y2 = np.minimum(bn[i, 3], bn[rem, 3])
                inter = np.maximum(x2 - x1, 0.0) * np.maximum(y2 - y1, 0.0)
                iou = inter / np.maximum(areas[i] + areas[rem] - inter, 1e-12)
                # Enclosing box diagonal^2
                ex1 = np.minimum(bn[i, 0], bn[rem, 0])
                ey1 = np.minimum(bn[i, 1], bn[rem, 1])
                ex2 = np.maximum(bn[i, 2], bn[rem, 2])
                ey2 = np.maximum(bn[i, 3], bn[rem, 3])
                c2 = (ex2 - ex1) ** 2 + (ey2 - ey1) ** 2 + 1e-12
                rho2 = (cx[i] - cx[rem]) ** 2 + (cy[i] - cy[rem]) ** 2
                diou = iou - (rho2 / c2) ** beta
                active[rem[diou > iou_thr]] = False
        kept_all = np.array(kept_all, dtype=np.int64)
        if len(kept_all) > max_det:
            kept_all = kept_all[np.argsort(-scores[kept_all])[:max_det]]
        for k in kept_all:
            x1, y1, x2, y2 = b_ori[k]
            preds.append((image_id, int(clses[k]), [float(x1), float(y1), float(x2 - x1), float(y2 - y1)], float(scores[k])))
    return preds


def run_soft_nms_gaussian(items, sigma=0.5, score_thr=0.001, max_det=300):
    preds = []
    for image_id, b_net, scores, clses, b_ori in items:
        if len(scores) == 0:
            continue
        out_boxes, out_scores, out_cls = [], [], []
        for c in np.unique(clses):
            m = np.flatnonzero(clses == c)
            # Limit per-class candidates to top 600 for fast exact Soft-NMS (score tail <0.005 irrelevant for AP)
            if len(m) > 600:
                m = m[:600]
            bn = b_net[m].copy()
            bo = b_ori[m].copy()
            sc = scores[m].copy()
            areas = np.maximum(bn[:, 2] - bn[:, 0], 0.0) * np.maximum(bn[:, 3] - bn[:, 1], 0.0)
            n = len(sc)
            for i in range(min(n, max_det)):
                idx = i + int(np.argmax(sc[i:]))
                if sc[idx] < score_thr:
                    break
                if idx != i:
                    bn[[i, idx]] = bn[[idx, i]]
                    bo[[i, idx]] = bo[[idx, i]]
                    sc[[i, idx]] = sc[[idx, i]]
                    areas[[i, idx]] = areas[[idx, i]]
                out_boxes.append(bo[i])
                out_scores.append(sc[i])
                out_cls.append(int(c))
                if i + 1 >= n:
                    break
                x1 = np.maximum(bn[i, 0], bn[i + 1:, 0])
                y1 = np.maximum(bn[i, 1], bn[i + 1:, 1])
                x2 = np.minimum(bn[i, 2], bn[i + 1:, 2])
                y2 = np.minimum(bn[i, 3], bn[i + 1:, 3])
                inter = np.maximum(x2 - x1, 0.0) * np.maximum(y2 - y1, 0.0)
                iou = inter / np.maximum(areas[i] + areas[i + 1:] - inter, 1e-12)
                sc[i + 1:] *= np.exp(-(iou * iou) / sigma)
        if not out_scores:
            continue
        out_scores = np.array(out_scores, dtype=np.float32)
        order = np.argsort(-out_scores)[:max_det]
        for k in order:
            x1, y1, x2, y2 = out_boxes[k]
            preds.append((image_id, out_cls[k], [float(x1), float(y1), float(x2 - x1), float(y2 - y1)], float(out_scores[k])))
    return preds


def run_box_voting(items, iou_thr=0.70, vote_iou=0.70, power=2.0, max_det=300):
    preds = []
    for image_id, b_net, scores, clses, b_ori in items:
        if len(scores) == 0:
            continue
        bt = torch.from_numpy(b_net)
        st = torch.from_numpy(scores)
        ct = torch.from_numpy(clses)
        offsets = ct.float() * 7680.0
        keep = torchvision.ops.nms(bt + offsets[:, None], st, iou_thr)[:max_det].numpy()
        # Refine each kept box using overlapping same-class candidates with IoU >= vote_iou
        for k in keep:
            c = clses[k]
            same = np.flatnonzero(clses == c)
            # Only consider candidates whose score >= 0.1 * kept score (avoid noisy tail dragging boxes)
            same = same[scores[same] >= max(0.01, 0.1 * scores[k])]
            if len(same) > 1:
                bn_k = b_net[k:k + 1]
                bn_s = b_net[same]
                ious = box_iou_matrix(bn_k, bn_s)[0]
                sel = same[ious >= vote_iou]
                if len(sel) > 1:
                    w = (scores[sel] ** power) * (ious[ious >= vote_iou])
                    w = w / np.maximum(w.sum(), 1e-12)
                    box = (b_ori[sel] * w[:, None]).sum(axis=0)
                else:
                    box = b_ori[k]
            else:
                box = b_ori[k]
            x1, y1, x2, y2 = box
            preds.append((image_id, int(c), [float(x1), float(y1), float(x2 - x1), float(y2 - y1)], float(scores[k])))
    return preds


def eval_preds(gt, cats, raw_preds):
    coco_preds = [
        dict(image_id=img_id, category_id=cats[reference.NAMES[c]], bbox=bbox, score=score)
        for img_id, c, bbox, score in raw_preds
    ]
    ev = reference.coco_evaluate(gt, coco_preds, [cats[n] for n in reference.NAMES])
    res = dict(
        AP50_95=reference.ap(ev),
        AP50=reference.ap(ev, iou=0.5),
        AP75=reference.ap(ev, iou=0.75),
        AR100=reference.ar100(ev),
        AP_small=reference.ap(ev, "small"),
        AP_medium=reference.ap(ev, "medium"),
        AP_large=reference.ap(ev, "large"),
    )
    return res


def main():
    print_training_stats()
    gt = COCO(str(reference.DATA / "annotations/instances_val.json"))
    img_ids = {Path(v["file_name"]).name: k for k, v in gt.imgs.items()}
    cats = {v["name"]: k for k, v in gt.cats.items()}

    methods = [
        ("std_0.65", lambda it: run_std_nms(it, 0.65)),
        ("std_0.70_official", lambda it: run_std_nms(it, 0.70)),
        ("std_0.75", lambda it: run_std_nms(it, 0.75)),
        ("std_0.80", lambda it: run_std_nms(it, 0.80)),
        ("diou_0.70", lambda it: run_diou_nms(it, 0.70)),
        ("diou_0.75", lambda it: run_diou_nms(it, 0.75)),
        ("soft_g0.3", lambda it: run_soft_nms_gaussian(it, 0.3)),
        ("soft_g0.5", lambda it: run_soft_nms_gaussian(it, 0.5)),
        ("soft_g0.7", lambda it: run_soft_nms_gaussian(it, 0.7)),
        ("vote_0.70_p1", lambda it: run_box_voting(it, 0.70, 0.70, 1.0)),
        ("vote_0.70_p2", lambda it: run_box_voting(it, 0.70, 0.70, 2.0)),
        ("vote_0.75_p2", lambda it: run_box_voting(it, 0.75, 0.75, 2.0)),
    ]

    out = {}
    print("\n=== ZERO-TRAIN NMS PROBE (RTX 4090, Val 1000 imgs, COCO maxDets=100) ===")
    for arm in ("C0", "HistoricalCLS", "H1_L1"):
        t0 = time.monotonic()
        ck = STAGE / "runs" / arm / "weights/best.pt"
        items = extract_raw(ck, gt, img_ids)
        official = json.loads((STAGE / "records" / arm / "evaluation/summary.json").read_text())["best"]["main"]
        print(f"\n--- {arm} (extract {time.monotonic()-t0:.1f}s) ---")
        print(f"{'Method':<18} | {'AP50:95':>8} {'d95':>7} | {'AP50':>8} {'d50':>7} | {'AP75':>8} {'d75':>7} | {'AR100':>7} | {'S':>7} {'M':>7} {'L':>7}")
        arm_res = {}
        base = None
        for name, fn in methods:
            m = eval_preds(gt, cats, fn(items))
            arm_res[name] = m
            if name == "std_0.70_official":
                base = m
                delta_chk = max(abs(m[k] - official[k]) for k in ("AP50_95", "AP50", "AP75"))
                print(f"  [replay check vs official summary.json: max_abs_delta={delta_chk:.6f}]")
                assert delta_chk < 1e-3, f"Replay mismatch on {arm}: {delta_chk} vs {official}"
        for name, m in arm_res.items():
            d95 = (m["AP50_95"] - base["AP50_95"]) * 100
            d50 = (m["AP50"] - base["AP50"]) * 100
            d75 = (m["AP75"] - base["AP75"]) * 100
            print(f"{name:<18} | {m['AP50_95']*100:8.3f} {d95:+7.3f} | {m['AP50']*100:8.3f} {d50:+7.3f} | {m['AP75']*100:8.3f} {d75:+7.3f} | {m['AR100']*100:7.3f} | {m['AP_small']*100:7.3f} {m['AP_medium']*100:7.3f} {m['AP_large']*100:7.3f}")
        out[arm] = arm_res
    dump(STAGE / "diagnostics/nms_probe_summary.json", out)
    print("\nSaved diagnostics/nms_probe_summary.json")


if __name__ == "__main__":
    main()

