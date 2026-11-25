"""Zero-retrain Paper 2 post-V7 diagnostics and unified RTX 4090 efficiency benchmark:
  1) Full 34,000-anchor diagnosis on b1_w50 best.pt & p2a_w50 best.pt (1024-space inverse-y pre-NMS)
  2) NMS coordinate system audit: Scheme 1 (Warp-space NMS -> inv-y) vs Scheme 2 (inv-y -> 1024 NMS, canonical)
  3) Unified Params, GFLOPs, pure GPU (bs=1, bs=32, FP32/FP16) & end-to-end (200 Val 1024x1024 imgs) latency.
"""
from collections import Counter
import copy
import json
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
from pycocotools.coco import COCO
import torch
from ultralytics.cfg import get_cfg
from ultralytics.data.augment import LetterBox
from ultralytics.utils import ops
from ultralytics.utils.nms import non_max_suppression
from ultralytics.utils.torch_utils import get_flops

STAGE = Path(__file__).resolve().parent
sys.path[:0] = [str(STAGE)]
import run_v2
import run_v6
import run_v7
import candidates_v6
import candidates_v7
import diagnose

reference = run_v2.reference


class WarpAudit(candidates_v6.InverseWarpValidatorMixin, diagnose.Audit):
    """34k anchor & @0.25 FP audit in original 1024 coords for Warp-trained models."""

    warp_split_y = 0.5
    warp_top_frac = 0.4

    def postprocess(self, pred):
        decoded, meta = pred
        assert not self.end2end and decoded.shape[1] == 11 and len(meta["feats"]) == 4
        y = decoded.detach().clone()
        batch = getattr(self, "_cur_warp_batch", None)
        assert batch is not None
        level = np.concatenate([np.full(int(v.shape[-2] * v.shape[-1]), k, np.int8) for k, v in enumerate(meta["feats"], 2)])
        assert len(level) == y.shape[2], (len(level), y.shape)

        bs = y.shape[0]
        raw_boxes_list = []
        raw_scores_list = []
        for si in range(bs):
            xywh_si = y[si, :4, :].transpose(0, 1)  # (34000, 4)
            xyxy_si = ops.xywh2xyxy(xywh_si)
            xyxy_orig = self._inverse_xyxy_to_orig(xyxy_si, batch["ori_shape"][si], batch["ratio_pad"][si])
            raw_boxes_list.append(xyxy_orig)
            y[si, :4, :] = ops.xyxy2xywh(xyxy_orig).transpose(0, 1)
            raw_scores_list.append(y[si, 4:, :].transpose(0, 1))

        detections, indices = non_max_suppression(
            y.clone(),
            0.001,
            0.7,
            multi_label=True,
            agnostic=False,
            nc=7,
            max_det=30000,
            max_time_img=5.0,
            return_idxs=True,
        )
        self.captured = (
            torch.stack(raw_boxes_list, 0),
            torch.stack(raw_scores_list, 0),
            level,
            detections,
            indices,
        )
        return [dict(bboxes=d[:300, :4], conf=d[:300, 4], cls=d[:300, 5], extra=d[:300, 6:]) for d in detections]

    def update_metrics(self, preds, batch):
        raw_boxes, raw_scores, level, dets, idxs = self.captured
        for si, name in enumerate(batch["im_file"]):
            pb = self._prepare_batch(si, batch)
            image_id = self.img_ids[Path(name).name]
            h, w = pb["ori_shape"]
            # raw_boxes[si] is ALREADY inverse-mapped to original 1024x1024 coords before NMS
            b = raw_boxes[si].cpu().numpy()
            scores = raw_scores[si].cpu().numpy()
            nms_boxes = b
            kin = idxs[si].detach().cpu().numpy().reshape(-1).astype(int)
            labels = dets[si][:, 5].detach().cpu().numpy().astype(int)
            if len(kin):
                assert np.max(np.abs(scores[kin, labels] - dets[si][:, 4].cpu().numpy())) < 1e-6
            pre_count = int(np.count_nonzero(scores > 0.001))
            real = self.gt.imgToAnns[image_id]
            truths = []
            for ann in real:
                if ann.get("iscrowd", 0) or ann.get("ignore", 0):
                    continue
                cn = self.gt.cats[ann["category_id"]]["name"]
                co = reference.NAMES.index(cn)
                x, y0, ww, hh = ann["bbox"]
                truth = np.array([x, y0, x + ww, y0 + hh], dtype=np.float32)
                gain = pb["ratio_pad"][0][0]
                row = dict(
                    image=Path(name).name,
                    image_id=image_id,
                    ann_id=ann["id"],
                    cls=cn,
                    short640=float(min(ww, hh) * gain),
                    area=ann.get("area", ww * hh),
                    inner=(min(x, y0, w - x - ww, h - y0 - hh) * gain) > 2.5,
                )
                row.update(diagnose.metrics_for_gt(truth, co, b, scores, level, kin, labels, pre_count, nms_boxes))
                self.rows.append(row)
                truths.append((co, truth))
            for pos in range(min(300, len(kin))):
                k = kin[pos]
                c = labels[pos]
                s = float(scores[k, c])
                if s < 0.25:
                    break
                same = max((float(diagnose.iou(b[k : k + 1], t)[0]) for kcls, t in truths if kcls == c), default=0.0)
                other = max((float(diagnose.iou(b[k : k + 1], t)[0]) for kcls, t in truths if kcls != c), default=0.0)
                self.detections.append(
                    dict(
                        image_id=image_id,
                        image=Path(name).name,
                        cls=reference.NAMES[c],
                        level=int(level[k]),
                        score=s,
                        iou_same=same,
                        iou_other=other,
                        box=b[k].tolist(),
                    )
                )
        self.captured = None
        super(diagnose.Audit, self).update_metrics(preds, batch)


def run_warp_34k_diagnosis(arm: str, split_y: float = 0.5, top_frac: float = 0.4) -> dict:
    ck = STAGE / "runs" / arm / "weights/best.pt"
    selected = json.loads((run_v2.proof(arm) / "evaluation/summary.json").read_text())["best"]
    assert run_v2.sha(ck) == selected["sha256"]
    dest = STAGE / "diagnostics" / arm
    folder = dest / "x0_y0"
    folder.mkdir(parents=True, exist_ok=True)

    gt = COCO(str(reference.DATA / "annotations/instances_val.json"))
    ids = {Path(v["file_name"]).name: k for k, v in gt.imgs.items()}
    cats = {v["name"]: k for k, v in gt.cats.items()}

    class _ConfiguredWarpAudit(WarpAudit):
        warp_split_y = float(split_y)
        warp_top_frac = float(top_frac)

    v = _ConfiguredWarpAudit(
        args=dict(
            task="detect",
            data=str(run_v2.DATA_YAML),
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
        save_dir=folder,
    )
    v.shift = (0, 0)
    v.gt = gt
    v.img_ids = ids
    v.rows = []
    v.detections = []
    v(model=reference.load_model(ck))
    assert v.seen == 1000 and len(v.rows) == 2612 and len({r["ann_id"] for r in v.rows}) == 2612

    base = v.rows
    result = dict(
        time=run_v2.now(),
        arm=arm,
        checkpoint_sha=run_v2.sha(ck),
        split="Val",
        warp=dict(split_y=split_y, top_frac=top_frac),
        groups={k: diagnose.group(g) for k, g in diagnose.groups(base).items()},
        detections_at_025={
            n: {
                "n": len(d := [r for r in v.detections if n == "all" or r["cls"] == n]),
                "background_like": sum(max(a["iou_same"], a["iou_other"]) < 0.10 for a in d),
                "wrong_class_overlap50": sum(a["iou_same"] < 0.5 and a["iou_other"] >= 0.5 for a in d),
            }
            for n in ["all", *reference.NAMES]
        },
    )
    diagnose.write_jsonl(folder / "detections_at_025.jsonl", v.detections)

    rows = [
        dict(image_id=ids[r["file_name"]], category_id=cats[reference.NAMES[int(r["category_id"]) - 1]], bbox=r["bbox"], score=r["score"])
        for r in v.jdict
    ]
    ev = reference.coco_evaluate(gt, rows, [cats[n] for n in reference.NAMES])
    result["one_to_one"] = diagnose.attach_coco_matches(ev, base)
    diagnose.write_jsonl(folder / "gt.jsonl", base)
    replay = dict(AP50=reference.ap(ev, iou=0.5), AP75=reference.ap(ev, iou=0.75), AP50_95=reference.ap(ev))
    delta = max(abs(val - selected["main"][k]) for k, val in replay.items())
    result["official_replay"] = dict(predictions=len(rows), metrics=replay, max_abs_metric_delta=delta)
    assert delta < 1e-4, result["official_replay"]
    result["complete"] = True
    run_v2.dump(dest / "summary.json", result)
    print(f"[34k Diag OK] {arm}: retained75={result['groups']['all']['stage75'].get('retained')}, raw_geo75={result['groups']['all']['stage75'].get('raw_geometry')}, replay_delta={delta:.2e}")
    return result


def eval_scheme1_warp_nms(ck: Path, split_y: float = 0.5, top_frac: float = 0.4, imgsz: int = 640) -> dict:
    """Scheme 1 control: run NMS in Warp pixel coordinates first, THEN inverse-map surviving boxes to 1024."""

    class _Scheme1Validator(candidates_v6.InverseWarpValidatorMixin, reference.TaskValidator):
        warp_split_y = float(split_y)
        warp_top_frac = float(top_frac)

        def postprocess(self, preds):
            if isinstance(preds, (list, tuple)):
                raw_pred = preds[0]
            else:
                raw_pred = preds
            # Run standard NMS BEFORE inverse mapping!
            from ultralytics.models.yolo.detect.val import DetectionValidator

            outs = DetectionValidator.postprocess(self, raw_pred)
            batch = getattr(self, "_cur_warp_batch", None)
            if batch is not None:
                for si, d in enumerate(outs):
                    if d["bboxes"].numel() > 0:
                        d["bboxes"] = self._inverse_xyxy_to_orig(d["bboxes"], batch["ori_shape"][si], batch["ratio_pad"][si])
            return outs

    with tempfile.TemporaryDirectory(prefix="scheme1_") as td:
        td_p = Path(td)
        model = reference.load_model(ck)
        v = _Scheme1Validator(
            args=dict(
                task="detect",
                data=str(run_v2.DATA_YAML),
                imgsz=imgsz,
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
            save_dir=td_p / "val",
        )
        v(model=model)
        gt = COCO(str(reference.DATA / "annotations/instances_val.json"))
        ids = {Path(x["file_name"]).stem: k for k, x in gt.imgs.items()}
        cats = {v0["name"]: k0 for k0, v0 in gt.cats.items()}
        preds = [
            dict(image_id=ids[Path(p["file_name"]).stem], category_id=cats[reference.NAMES[int(p["category_id"]) - 1]], bbox=p["bbox"], score=p["score"])
            for p in v.jdict
        ]
        ev = reference.coco_evaluate(gt, preds, [cats[n] for n in reference.NAMES])
        return dict(
            predictions=len(preds),
            AP50=reference.ap(ev, iou=0.5),
            AP75=reference.ap(ev, iou=0.75),
            AP50_95=reference.ap(ev),
            AR100=reference.ar100(ev),
        )


def benchmark_model_efficiency(spec: dict, val_images: list[np.ndarray]) -> dict:
    """Measure params, GFLOPs, pure GPU (bs=1/32, FP32/FP16), and 200-img Val 1024->eval end-to-end latency."""
    name = spec["name"]
    ck = spec["ck"]
    imgsz = spec["imgsz"]
    wcfg = spec["warp"]
    mode = spec["mode"]  # "o2m" or "o2o"

    raw_m = reference.load_model(ck)
    train_params = sum(p.numel() for p in raw_m.parameters())
    if mode == "o2o":
        dep = copy.deepcopy(raw_m).float().eval()
        dep.fuse(verbose=False)
    else:
        dep = candidates_v7.compact_o2m(raw_m)
    deploy_params = sum(p.numel() for p in dep.parameters())
    flops_g = float(get_flops(copy.deepcopy(dep).cpu(), imgsz=imgsz))

    dep_fp32 = copy.deepcopy(dep).cuda().eval()
    dep_fp16 = copy.deepcopy(dep).cuda().half().eval()

    def _pure_gpu_ms(model, bs: int, dtype) -> float:
        x = torch.randn(bs, 3, imgsz, imgsz, device="cuda", dtype=dtype)
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

    # Real 200-image 1024x1024 end-to-end pipeline timing (preprocess/warp -> H2D -> forward -> inv-y -> NMS/decode -> CPU)
    lb = LetterBox((imgsz, imgsz), auto=False, stride=32)
    t_pre = t_fwd = t_post = 0.0
    with torch.no_grad():
        for im0 in val_images[:20]:
            im_p = candidates_v6.warp_image(im0, imgsz, wcfg[0], wcfg[1]) if wcfg else cv2.resize(im0, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
            x = torch.from_numpy(np.ascontiguousarray(im_p[:, :, ::-1].transpose(2, 0, 1))).unsqueeze(0).cuda().float() / 255.0
            _ = dep_fp32(x)
        torch.cuda.synchronize()

        t_total0 = time.perf_counter()
        for im0 in val_images:
            tp0 = time.perf_counter()
            if wcfg is not None:
                im_p = candidates_v6.warp_image(im0, imgsz, wcfg[0], wcfg[1])
            else:
                im_p = cv2.resize(im0, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
            t_in = np.ascontiguousarray(im_p[:, :, ::-1].transpose(2, 0, 1))
            x = torch.from_numpy(t_in).unsqueeze(0).cuda(non_blocking=False).float() * (1.0 / 255.0)
            torch.cuda.synchronize()
            tp1 = time.perf_counter()
            out = dep_fp32(x)
            torch.cuda.synchronize()
            tp2 = time.perf_counter()
            pred = out[0].clone()
            if mode == "o2o":
                if wcfg is not None:
                    mask = pred[0, :, 4] > 0
                    if mask.any():
                        xyxy = pred[0, mask, :4].clone()
                        _, _, eff_top = candidates_v6.effective_warp_frac(imgsz, wcfg[0], wcfg[1])
                        xyxy[:, 0] *= 1024.0 / imgsz
                        xyxy[:, 2] *= 1024.0 / imgsz
                        xyxy[:, 1] = candidates_v6.warp_y_inverse_torch(xyxy[:, 1] / imgsz, wcfg[0], eff_top) * 1024.0
                        xyxy[:, 3] = candidates_v6.warp_y_inverse_torch(xyxy[:, 3] / imgsz, wcfg[0], eff_top) * 1024.0
                        pred[0, mask, :4] = xyxy
                dets = non_max_suppression(pred, 0.25, 0.70, max_det=300, end2end=True)
            else:
                if wcfg is not None:
                    xyxy = ops.xywh2xyxy(pred[0, :4].transpose(0, 1))
                    y1n, y2n = xyxy[:, 1] / imgsz, xyxy[:, 3] / imgsz
                    xyxy[:, 0] *= 1024.0 / imgsz
                    xyxy[:, 2] *= 1024.0 / imgsz
                    xyxy[:, 1] = candidates_v6.warp_y_inverse_torch(y1n, wcfg[0], wcfg[1]) * 1024.0
                    xyxy[:, 3] = candidates_v6.warp_y_inverse_torch(y2n, wcfg[0], wcfg[1]) * 1024.0
                    pred[0, :4] = ops.xyxy2xywh(xyxy).transpose(0, 1)
                dets = non_max_suppression(pred, 0.25, 0.70, multi_label=True, agnostic=False, nc=7, max_det=300)
            _ = dets[0].cpu()
            tp3 = time.perf_counter()
            t_pre += tp1 - tp0
            t_fwd += tp2 - tp1
            t_post += tp3 - tp2
        e2e_ms = (time.perf_counter() - t_total0) * 1000.0 / len(val_images)

    del dep_fp32, dep_fp16
    torch.cuda.empty_cache()

    res = dict(
        name=name,
        mode=mode,
        imgsz=imgsz,
        warp=wcfg,
        train_params=train_params,
        deploy_params=deploy_params,
        gflops=round(flops_g, 3),
        gpu_fp32_bs1_ms=round(gpu_fp32_bs1, 3),
        gpu_fp32_bs32_ms=round(gpu_fp32_bs32, 3),
        gpu_fp16_bs1_ms=round(gpu_fp16_bs1, 3),
        gpu_fp16_bs32_ms=round(gpu_fp16_bs32, 3),
        e2e_bs1_ms=round(e2e_ms, 3),
        e2e_bs1_fps=round(1000.0 / e2e_ms, 2),
        e2e_pre_ms=round(t_pre * 1000.0 / len(val_images), 3),
        e2e_fwd_ms=round(t_fwd * 1000.0 / len(val_images), 3),
        e2e_post_ms=round(t_post * 1000.0 / len(val_images), 3),
    )
    print(f"  [Bench] {name:<18}: params={train_params}/{deploy_params}, GFLOPs={res['gflops']}, GPU32={res['gpu_fp32_bs32_ms']:.3f}ms, E2E1={res['e2e_bs1_ms']:.3f}ms ({res['e2e_bs1_fps']} FPS)")
    return res


def main():
    print("================ V7 ZERO-RETRAIN DIAGNOSTICS & BENCHMARK ================")
    diag_out = {}
    for arm in ("p2a_w50", "b1_w50"):
        diag_out[arm] = run_warp_34k_diagnosis(arm, 0.5, 0.4)

    nms_audit = {}
    for arm, wcfg in (("p2a_w50", (0.5, 0.4)), ("b1_w50", (0.5, 0.4)), ("b0_w50", (0.5, 0.4)), ("b1_w448", (0.5, 0.3))):
        ck = STAGE / "runs" / arm / "weights/best.pt"
        if ck.is_file():
            s2 = json.loads((run_v2.proof(arm) / "evaluation/summary.json").read_text())["best"]["main"]
            s1 = eval_scheme1_warp_nms(ck, wcfg[0], wcfg[1], 640)
            nms_audit[arm] = dict(
                scheme2_1024_nms={k: round(s2[k] * 100, 3) for k in ("AP50", "AP75", "AP50_95", "AR100")},
                scheme1_warp_nms={k: round(s1[k] * 100, 3) for k in ("AP50", "AP75", "AP50_95", "AR100")},
                delta_s2_minus_s1_pp={k: round((s2[k] - s1[k]) * 100, 3) for k in ("AP50", "AP75", "AP50_95", "AR100")},
            )
            print(f"  [NMS Audit {arm}] S2(1024)={nms_audit[arm]['scheme2_1024_nms']['AP50_95']} vs S1(warp)={nms_audit[arm]['scheme1_warp_nms']['AP50_95']}")
    run_v2.dump(STAGE / "diagnostics/v7_nms_coord_audit.json", nms_audit)

    val_files = (run_v2.REF / "data/val.txt").read_text().splitlines()[:200]
    val_imgs = [cv2.imread(p) for p in val_files]
    specs = [
        dict(name="b0_o2o", ck=STAGE / "runs/b0/weights/best.pt", imgsz=640, warp=None, mode="o2o"),
        dict(name="b0_o2m", ck=STAGE / "runs/b0/weights/best.pt", imgsz=640, warp=None, mode="o2m"),
        dict(name="b0_w50", ck=STAGE / "runs/b0_w50/weights/best.pt", imgsz=640, warp=(0.5, 0.4), mode="o2m"),
        dict(name="b0_p2", ck=STAGE / "runs/b0_p2/weights/best.pt", imgsz=640, warp=None, mode="o2m"),
        dict(name="HistCLS", ck=STAGE / "runs/HistoricalCLS_DualBest/weights/best.pt", imgsz=640, warp=None, mode="o2m"),
        dict(name="CLS_P2A", ck=STAGE / "runs/HistCLS_SchemeA_P2/weights/best.pt", imgsz=640, warp=None, mode="o2m"),
        dict(name="b1", ck=STAGE / "runs/b1/weights/best.pt", imgsz=640, warp=None, mode="o2m"),
        dict(name="p2a_w50", ck=STAGE / "runs/p2a_w50/weights/best.pt", imgsz=640, warp=(0.5, 0.4), mode="o2m"),
        dict(name="b1_w50", ck=STAGE / "runs/b1_w50/weights/best.pt", imgsz=640, warp=(0.5, 0.4), mode="o2m"),
        dict(name="b1_768", ck=STAGE / "runs/b1_768/weights/best.pt", imgsz=768, warp=None, mode="o2m"),
        dict(name="b1_w448", ck=STAGE / "runs/b1_w448/weights/best.pt", imgsz=640, warp=(0.5, 0.3), mode="o2m"),
    ]
    bench = [benchmark_model_efficiency(s, val_imgs) for s in specs if Path(s["ck"]).is_file()]
    run_v2.dump(STAGE / "diagnostics/v7_latency_and_flops.json", dict(time=run_v2.now(), gpu=torch.cuda.get_device_name(0), n_e2e_imgs=len(val_imgs), models=bench))
    print("================ V7 ZERO-RETRAIN BENCH & DIAG COMPLETED ================")


if __name__ == "__main__":
    main()
