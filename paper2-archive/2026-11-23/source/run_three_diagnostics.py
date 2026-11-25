"""Run three fast RTX 4090 diagnostics:
1) Counterfactual q_delta ON vs OFF on HistCLS_O2M_RegQual (best.pt E76 & best_ap50.pt E77)
2) Full SVRDD7 Train-set (6,000 imgs / 15,619 GTs) TAL positive assignment audit on HistoricalCLS (E50 & E86)
3) Real RTX 4090 inference latency benchmark (HistoricalCLS vs HistCLS_AntiAlias_L01 vs HistCLS_O2M_RegQual)
"""
import contextlib
import copy
import io
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from pycocotools.coco import COCO

STAGE = Path("/root/YOLO26N-FCHL/reports/paper2_cls_candidate_diagnosis_640_20261120_42106")
GEO = Path("/root/YOLO26N-FCHL/reports/paper2_geometry_semantic_640_20261109_42266")
REF = Path("/root/YOLO26N-FCHL/reports/paper2_cls_p2p3_snake_640_20261115_42264")
sys.path[:0] = [
    str(STAGE),
    str(GEO / "source"),
    str(REF / "source/dependencies"),
    str(GEO / "source/dependencies"),
    "/root/YOLO26N-FCHL",
]

import candidates_v2
import historical_head
import models
import experiment as reference
from run import DATA_YAML, dump
from diagnose import Audit, groups, group
from ultralytics.utils.metrics import bbox_iou
from ultralytics.utils.tal import make_anchors, dist2bbox


def run_qdelta_counterfactual():
    print("\n================ 1. COUNTERFACTUAL q_delta ON vs OFF (HistCLS_O2M_RegQual) ================")
    gt = COCO(str(reference.DATA / "annotations/instances_val.json"))
    img_ids = {Path(v["file_name"]).name: k for k, v in gt.imgs.items()}
    cats = {v["name"]: k for k, v in gt.cats.items()}

    out = {}
    for ck_name in ("best", "best_ap50"):
        ck_path = STAGE / "runs/HistCLS_O2M_RegQual/weights" / f"{ck_name}.pt"
        arm_res = {}
        for mode in ("ON", "OFF"):
            model = reference.load_model(ck_path)
            head = model.model[-1]
            # Collect q_delta stats if ON; zero out final Conv1x1 if OFF (boxes are 100% invariant!)
            q_stats = []
            if mode == "OFF":
                with torch.no_grad():
                    for m in head.reg_qual_o2m:
                        m[-1].weight.zero_()
                        m[-1].bias.zero_()
            else:
                # Register hooks to measure q_delta magnitudes on first 10 batches
                hooks = []
                def make_hook(idx):
                    def _hk(_mod, _inp, output):
                        if len(q_stats) < 40:
                            q = output.detach().float()
                            q_stats.append((idx, float(q.mean()), float(q.std()), float(q.min()), float(q.max())))
                    return _hk
                for i, m in enumerate(head.reg_qual_o2m):
                    hooks.append(m.register_forward_hook(make_hook(i)))

            folder = STAGE / f"diagnostics/_tmp_qdelta_{ck_name}_{mode}"
            folder.mkdir(parents=True, exist_ok=True)
            v = Audit(
                args=dict(
                    task="detect", data=str(DATA_YAML), imgsz=640, batch=32, workers=4,
                    device="0", conf=0.001, iou=0.7, max_det=300, rect=True,
                    save_json=True, plots=False, verbose=False, half=False, split="val",
                ),
                save_dir=folder,
            )
            v.shift = (0, 0)
            v.gt = gt
            v.img_ids = img_ids
            v.rows = []
            v.detections = []
            with contextlib.redirect_stdout(io.StringIO()):
                v(model=model)
            if mode == "ON":
                for h in hooks:
                    h.remove()

            preds = [
                dict(
                    image_id=img_ids[r["file_name"]],
                    category_id=cats[reference.NAMES[int(r["category_id"]) - 1]],
                    bbox=r["bbox"],
                    score=r["score"],
                )
                for r in v.jdict
            ]
            ev = reference.coco_evaluate(gt, preds, [cats[n] for n in reference.NAMES])
            main_m = dict(
                AP50_95=reference.ap(ev),
                AP50=reference.ap(ev, iou=0.5),
                AP75=reference.ap(ev, iou=0.75),
                AR100=reference.ar100(ev),
                AP_small=reference.ap(ev, "small"),
                AP_medium=reference.ap(ev, "medium"),
                AP_large=reference.ap(ev, "large"),
            )
            cls_m = {}
            for n in reference.NAMES:
                evc = reference.coco_evaluate(gt, preds, [cats[n]])
                cls_m[n] = dict(AP50_95=reference.ap(evc), AP50=reference.ap(evc, iou=0.5), AP75=reference.ap(evc, iou=0.75))

            grp = {k: group(g) for k, g in groups(v.rows).items()}
            det_summary = {}
            for n in ("all", *reference.NAMES):
                d = [r for r in v.detections if n == "all" or r["cls"] == n]
                det_summary[n] = dict(
                    n=len(d),
                    bg=sum(max(a["iou_same"], a["iou_other"]) < 0.10 for a in d),
                    wrong_cls=sum(a["iou_same"] < 0.50 and a["iou_other"] >= 0.50 for a in d),
                )
            # Check witness IoU medians on NMS-suppressed GTs
            nms_rows = [r for r in v.rows if r["stage75"] == "nms" and "overlap_witness_iou_gt75" in r]
            nms_stats = dict(
                nms_count=len(nms_rows),
                qual_iou_med=float(np.median([r["score_best_iou75"] for r in nms_rows])) if nms_rows else 0.0,
                win_iou_med=float(np.median([r["overlap_witness_iou_gt75"] for r in nms_rows])) if nms_rows else 0.0,
            )
            # Keep per-GT map for direct ON vs OFF flip analysis
            arm_res[mode] = dict(
                main=main_m,
                classes=cls_m,
                stage75_all=grp["all"]["stage75"],
                stage75_narrow8=grp["narrow_lt8"]["stage75"],
                stage75_lctc16=grp["LC_TC_lt16"]["stage75"],
                nms_stats=nms_stats,
                det_025=det_summary,
                raw_rows={r["ann_id"]: (r["stage75"], r["raw_max_iou"], r.get("score75")) for r in v.rows},
            )
            if mode == "ON" and q_stats:
                by_lvl = defaultdict(list)
                for idx, mn, sd, lo, hi in q_stats:
                    by_lvl[idx].append((mn, sd, lo, hi))
                arm_res["q_delta_stats"] = {
                    f"P{idx+2}": dict(
                        mean=float(np.mean([x[0] for x in vals])),
                        std=float(np.mean([x[1] for x in vals])),
                        min=float(np.min([x[2] for x in vals])),
                        max=float(np.max([x[3] for x in vals])),
                    )
                    for idx, vals in sorted(by_lvl.items())
                }

        # Direct exact-same-box flip comparison ON vs OFF
        r_on, r_off = arm_res["ON"]["raw_rows"], arm_res["OFF"]["raw_rows"]
        max_raw_iou_diff = max(abs(r_on[k][1] - r_off[k][1]) for k in r_on)
        assert max_raw_iou_diff < 1e-6, f"Boxes changed between ON and OFF! {max_raw_iou_diff}"
        rescued_from_nms = sum(r_off[k][0] == "nms" and r_on[k][0] == "retained" for k in r_on)
        lost_to_nms = sum(r_off[k][0] == "retained" and r_on[k][0] == "nms" for k in r_on)
        rescued_from_flr = sum(r_off[k][0] == "correct_class_below_floor" and r_on[k][0] == "retained" for k in r_on)
        lost_to_flr = sum(r_off[k][0] == "retained" and r_on[k][0] == "correct_class_below_floor" for k in r_on)
        arm_res["causal_flips"] = dict(
            max_raw_iou_diff=max_raw_iou_diff,
            rescued_from_nms=rescued_from_nms,
            lost_to_nms=lost_to_nms,
            net_nms_rescue=rescued_from_nms - lost_to_nms,
            rescued_from_floor=rescued_from_flr,
            lost_to_floor=lost_to_flr,
        )
        # Drop large raw_rows before saving JSON
        del arm_res["ON"]["raw_rows"], arm_res["OFF"]["raw_rows"]
        out[ck_name] = arm_res

        m_on, m_off = arm_res["ON"]["main"], arm_res["OFF"]["main"]
        print(f"\n--- HistCLS_O2M_RegQual [{ck_name}.pt] (box_diff={max_raw_iou_diff:.2e}) ---")
        print(f"  q_delta OFF: AP50:95={m_off['AP50_95']*100:.3f}%, AP50={m_off['AP50']*100:.3f}%, AP75={m_off['AP75']*100:.3f}%, AR100={m_off['AR100']*100:.3f}%, S={m_off['AP_small']*100:.3f}%, M={m_off['AP_medium']*100:.3f}%, L={m_off['AP_large']*100:.3f}%")
        print(f"  q_delta ON : AP50:95={m_on['AP50_95']*100:.3f}% ({(m_on['AP50_95']-m_off['AP50_95'])*100:+.3f}), AP50={m_on['AP50']*100:.3f}% ({(m_on['AP50']-m_off['AP50'])*100:+.3f}), AP75={m_on['AP75']*100:.3f}% ({(m_on['AP75']-m_off['AP75'])*100:+.3f}), AR100={m_on['AR100']*100:.3f}%")
        print(f"  stage75 OFF: {arm_res['OFF']['stage75_all']} | det@0.25={arm_res['OFF']['det_025']['all']} TP_wc={arm_res['OFF']['det_025']['TP']['wrong_cls']}")
        print(f"  stage75 ON : {arm_res['ON']['stage75_all']} | det@0.25={arm_res['ON']['det_025']['all']} TP_wc={arm_res['ON']['det_025']['TP']['wrong_cls']}")
        print(f"  causal flips (ON vs OFF on IDENTICAL boxes): {arm_res['causal_flips']}")
        print(f"  q_delta stats: {arm_res.get('q_delta_stats')}")
        cls_deltas_95 = ", ".join(f"{n}:{(arm_res['ON']['classes'][n]['AP50_95']-arm_res['OFF']['classes'][n]['AP50_95'])*100:+.2f}" for n in reference.NAMES)
        cls_deltas_50 = ", ".join(f"{n}:{(arm_res['ON']['classes'][n]['AP50']-arm_res['OFF']['classes'][n]['AP50'])*100:+.2f}" for n in reference.NAMES)
        print(f"  7-cls ON-OFF dAP50:95: {cls_deltas_95}")
        print(f"  7-cls ON-OFF dAP50   : {cls_deltas_50}")

    dump(STAGE / "diagnostics/qdelta_counterfactual_summary.json", out)
    return out


def run_tal_assignment_audit():
    print("\n================ 2. TRAINING-SET TAL POSITIVE ASSIGNMENT AUDIT (SVRDD7 Train 6,000 imgs) ================")
    from ultralytics.data.build import build_yolo_dataset, build_dataloader
    from ultralytics.cfg import get_cfg

    train_txt = str(REF / "data/train.txt")
    cfg = get_cfg(overrides=dict(
        imgsz=640, batch=32, rect=False, mosaic=0.0, mixup=0.0, cutmix=0.0, copy_paste=0.0,
        degrees=0.0, translate=0.0, scale=0.0, shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.0,
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.0, erasing=0.0, auto_augment=None, bgr=0.0,
        single_cls=False, workers=4, cache=False,
    ))
    data_dict = dict(names=dict(enumerate(reference.NAMES)), nc=7, channels=3)
    ds = build_yolo_dataset(cfg, train_txt, batch=32, data=data_dict, mode="val", rect=False, stride=32)
    dl = build_dataloader(ds, batch=32, workers=4, shuffle=False, rank=-1)
    assert len(ds) == 6000

    out = {}
    for ck_label, ck_file in [
        ("HistoricalCLS_E50", STAGE / "runs/HistoricalCLS/weights/epoch50.pt"),
        ("HistoricalCLS_E86_best", STAGE / "runs/HistoricalCLS/weights/best.pt"),
    ]:
        t0 = time.monotonic()
        ck = torch.load(ck_file, map_location="cpu", weights_only=False)
        model = copy.deepcopy(ck.get("ema") or ck["model"]).float().cuda().eval()
        crit = model.init_criterion()
        for b in (crit.one2many, crit.one2one):
            b.assigner.stride = [8.0, 16.0, 32.0]
        o2m_loss = crit.one2many
        assigner = o2m_loss.assigner

        records = []
        with torch.no_grad():
            for batch in dl:
                imgs = batch["img"].cuda().float() / 255.0
                # Forward through eval model to get raw O2M preds dict (boxes, scores, feats)
                _, preds_dict = model(imgs)
                preds = preds_dict["one2many"]
                pred_distri = preds["boxes"].permute(0, 2, 1).contiguous()
                pred_scores = preds["scores"].permute(0, 2, 1).contiguous()
                feats = preds["feats"]
                anchor_points, stride_tensor = make_anchors(feats, o2m_loss.stride.to(imgs.device), 0.5)
                lvl_sizes = [f.shape[2] * f.shape[3] for f in feats]
                lvl_id = torch.cat([torch.full((sz,), k, device=imgs.device, dtype=torch.long) for k, sz in enumerate(lvl_sizes, 2)])

                bs = pred_scores.shape[0]
                imgsz = torch.tensor(feats[0].shape[2:], device=imgs.device, dtype=pred_scores.dtype) * o2m_loss.stride[0].to(imgs.device)
                targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
                targets = o2m_loss.preprocess(targets.to(imgs.device), bs, scale_tensor=imgsz[[1, 0, 1, 0]])
                gt_labels, gt_bboxes = targets.split((1, 4), 2)
                mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)
                if not mask_gt.any():
                    continue
                pred_bboxes_stride = o2m_loss.bbox_decode(anchor_points, pred_distri)
                pred_bboxes_px = (pred_bboxes_stride * stride_tensor).type(gt_bboxes.dtype)
                pd_probs = pred_scores.sigmoid()

                # Replicate TaskAlignedAssigner._forward explicitly so we capture per-GT stats BEFORE multi-GT resolution AND AFTER!
                assigner.bs = bs
                assigner.n_max_boxes = gt_bboxes.shape[1]
                mask_pos, align_metric, overlaps = assigner.get_pos_mask(
                    pd_probs, pred_bboxes_px, gt_labels, gt_bboxes, anchor_points * stride_tensor, mask_gt
                )
                target_gt_idx, fg_mask, mask_pos_res = assigner.select_highest_overlaps(
                    mask_pos, overlaps, assigner.n_max_boxes, align_metric
                )
                # Normalized target scores per GT
                am_pos = align_metric * mask_pos_res
                pos_align_max = am_pos.amax(dim=-1, keepdim=True)
                pos_ov_max = (overlaps * mask_pos_res).amax(dim=-1, keepdim=True)
                norm_per_gt = am_pos * pos_ov_max / (pos_align_max + assigner.eps)  # (bs, n_max, na)

                # Extract per-GT metrics
                m_gt_cpu = mask_gt.squeeze(-1).cpu().numpy()
                gt_l_cpu = gt_labels.squeeze(-1).long().cpu().numpy()
                gt_b_cpu = gt_bboxes.cpu().numpy()
                # Compute true IoU (not CIoU) for assigned positives only (fast!)
                for bi in range(bs):
                    valid_js = np.flatnonzero(m_gt_cpu[bi])
                    for ji in valid_js:
                        cls_idx = int(gt_l_cpu[bi, ji])
                        x1, y1, x2, y2 = gt_b_cpu[bi, ji]
                        short_px = float(min(x2 - x1, y2 - y1))
                        long_px = float(max(x2 - x1, y2 - y1))
                        pos_idx = mask_pos_res[bi, ji].nonzero(as_tuple=False).squeeze(-1)
                        n_pos = int(pos_idx.numel())
                        if n_pos > 0:
                            pb_sel = pred_bboxes_px[bi, pos_idx]
                            gb_sel = gt_bboxes[bi, ji : ji + 1].expand(n_pos, -1)
                            true_ious = bbox_iou(gb_sel, pb_sel, xywh=False, CIoU=False).squeeze(-1)
                            max_iou = float(true_ious.max().item())
                            mean_iou = float(true_ious.mean().item())
                            max_ciou = float(overlaps[bi, ji, pos_idx].max().item())
                            t_sc = norm_per_gt[bi, ji, pos_idx]
                            max_tscore = float(t_sc.max().item())
                            sum_tscore = float(t_sc.sum().item())
                            p_sc = pd_probs[bi, pos_idx, cls_idx]
                            max_pscore = float(p_sc.max().item())
                            lvls = lvl_id[pos_idx].cpu().tolist()
                            lvl_counts = [lvls.count(2), lvls.count(3), lvls.count(4), lvls.count(5)]
                        else:
                            max_iou = mean_iou = max_ciou = max_tscore = sum_tscore = max_pscore = 0.0
                            lvl_counts = [0, 0, 0, 0]
                        records.append((cls_idx, short_px, long_px, n_pos, max_iou, mean_iou, max_ciou, max_tscore, sum_tscore, max_pscore, lvl_counts))

        print(f"\n--- {ck_label}: audited {len(records)} Train GTs in {time.monotonic()-t0:.1f}s ---")
        def summarize(sub):
            if not sub:
                return {"n": 0}
            n = len(sub)
            npos = np.array([r[3] for r in sub])
            miou = np.array([r[4] for r in sub])
            mciou = np.array([r[6] for r in sub])
            mtsc = np.array([r[7] for r in sub])
            stsc = np.array([r[8] for r in sub])
            mpsc = np.array([r[9] for r in sub])
            lvls = np.array([r[10] for r in sub]).sum(axis=0)
            tot_pos = max(int(lvls.sum()), 1)
            return dict(
                n=n,
                pos_mean=float(npos.mean()),
                pos_med=float(np.median(npos)),
                pct_zero_pos=float((npos == 0).mean() * 100),
                pct_lt3_pos=float((npos < 3).mean() * 100),
                max_iou_med=float(np.median(miou)),
                max_ciou_med=float(np.median(mciou)),
                max_tscore_med=float(np.median(mtsc)),
                sum_tscore_med=float(np.median(stsc)),
                max_pscore_med=float(np.median(mpsc)),
                lvl_share_pct={f"P{k}": round(float(lvls[k - 2]) / tot_pos * 100, 1) for k in (2, 3, 4, 5)},
            )

        buckets = [
            ("ALL_short<8", [r for r in records if r[1] < 8]),
            ("ALL_8<=s<16", [r for r in records if 8 <= r[1] < 16]),
            ("ALL_16<=s<32", [r for r in records if 16 <= r[1] < 32]),
            ("ALL_short>=32", [r for r in records if r[1] >= 32]),
            ("LC_short<16", [r for r in records if r[0] == 0 and r[1] < 16]),
            ("LC_short>=32", [r for r in records if r[0] == 0 and r[1] >= 32]),
            ("TC_short<16", [r for r in records if r[0] == 1 and r[1] < 16]),
            ("TC_short>=32", [r for r in records if r[0] == 1 and r[1] >= 32]),
            ("P_short<16", [r for r in records if r[0] == 3 and r[1] < 16]),
            ("MC_short<16", [r for r in records if r[0] == 4 and r[1] < 16]),
        ]
        ck_out = {}
        print(f"{'Bucket':<14} | {'N':>5} | {'pos':>5} {'0%':>5} {'<3%':>5} | {'mIoU':>5} {'mCIoU':>5} | {'tScore':>6} {'sumT':>5} | {'pScore':>6} | {'P2/P3/P4/P5 %'}")
        for bname, sub in buckets:
            st = summarize(sub)
            ck_out[bname] = st
            ls = st["lvl_share_pct"]
            lstr = f"{ls['P2']:4.1f}/{ls['P3']:4.1f}/{ls['P4']:4.1f}/{ls['P5']:4.1f}"
            print(f"{bname:<14} | {st['n']:>5} | {st['pos_mean']:5.2f} {st['pct_zero_pos']:5.1f} {st['pct_lt3_pos']:5.1f} | {st['max_iou_med']:5.3f} {st['max_ciou_med']:5.3f} | {st['max_tscore_med']:6.3f} {st['sum_tscore_med']:5.2f} | {st['max_pscore_med']:6.3f} | {lstr}")
        out[ck_label] = ck_out

    dump(STAGE / "diagnostics/tal_train_assignment_audit.json", out)
    return out


def fold_antialias_to_5x5(aa_mod):
    """Algebraically fold (1-a)*Conv3x3_s2 + a*Blur3x3_s2(Conv3x3_s1) into a single Conv5x5_s2 (interior exact)."""
    aa = copy.deepcopy(aa_mod).eval().fuse_bn_eval()
    w = aa.conv.weight.detach().float()  # (co, ci, 3, 3)
    b = aa.conv.bias.detach().float()    # (co,)
    a = aa.alpha.detach().float().view(-1, 1, 1, 1)  # (co, 1, 1, 1)
    k1 = torch.tensor([1.0, 2.0, 1.0], dtype=torch.float32) / 4.0
    blur = torch.outer(k1, k1).view(1, 1, 3, 3)  # symmetric
    # 2D discrete conv of w (3x3) with blur (3x3) -> 5x5
    co, ci = w.shape[:2]
    w_pad = F.pad(w, (2, 2, 2, 2))  # (co, ci, 7, 7)
    w_blur = F.conv2d(w_pad.view(co * ci, 1, 7, 7), blur).view(co, ci, 5, 5)
    w_s2_5x5 = F.pad(w, (1, 1, 1, 1))
    w_eff = (1.0 - a) * w_s2_5x5 + a * w_blur
    from ultralytics.nn.modules import Conv
    out = Conv(ci, co, 5, 2, p=2)
    out.conv = torch.nn.Conv2d(ci, co, 5, stride=2, padding=2, bias=True)
    out.conv.weight.data.copy_(w_eff)
    out.conv.bias.data.copy_(b)
    out.act = copy.deepcopy(aa.act)
    del out.bn
    out.forward = out.forward_fuse
    out.i, out.f, out.type = aa.i, aa.f, "FoldedAntiAlias5x5"
    return out.eval()


def run_latency_benchmark():
    print("\n================ 3. RTX 4090 DEPLOYMENT LATENCY BENCHMARK (O2M FP32) ================")
    models_to_test = {}
    for name, ck_path in [
        ("HistoricalCLS", STAGE / "runs/HistoricalCLS/weights/best.pt"),
        ("HistCLS_O2M_RegQual", STAGE / "runs/HistCLS_O2M_RegQual/weights/best.pt"),
        ("HistCLS_AntiAlias_L01_unfolded", STAGE / "runs/HistCLS_AntiAlias_L01/weights/best.pt"),
    ]:
        m = reference.load_model(ck_path)
        m = candidates_v2.compact_o2m(m).cuda().eval()
        models_to_test[name] = m

    # Also build 5x5 folded version of HistCLS_AntiAlias_L01
    m_fold = copy.deepcopy(models_to_test["HistCLS_AntiAlias_L01_unfolded"]).cpu().eval()
    m_fold.model[0] = fold_antialias_to_5x5(m_fold.model[0])
    m_fold.model[1] = fold_antialias_to_5x5(m_fold.model[1])
    models_to_test["HistCLS_AntiAlias_L01_folded5x5"] = m_fold.cuda().eval()

    # Verify interior output parity of 5x5 fold on 640x640 image
    with torch.no_grad():
        x_chk = torch.rand(1, 3, 640, 640, device="cuda")
        y_u = models_to_test["HistCLS_AntiAlias_L01_unfolded"](x_chk)[0]
        y_f = models_to_test["HistCLS_AntiAlias_L01_folded5x5"](x_chk)[0]
        print(f"  5x5 fold full-img max diff (includes 1px border reflect vs zero): boxes={float((y_u[:,:4]-y_f[:,:4]).abs().max()):.4f}px, scores={float((y_u[:,4:]-y_f[:,4:]).abs().max()):.5f}, median_score_diff={float((y_u[:,4:]-y_f[:,4:]).abs().median()):.2e}")

    res = {}
    for bs in (1, 32):
        x = torch.rand(bs, 3, 640, 640, device="cuda")
        print(f"\n  --- Batch={bs} (640x640 FP32, 50 warmup + 200 iters) ---")
        for name, m in models_to_test.items():
            params = sum(p.numel() for p in m.parameters())
            with torch.no_grad():
                for _ in range(50):
                    _ = m(x)
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                iters = 200 if bs == 1 else 60
                for _ in range(iters):
                    _ = m(x)
                torch.cuda.synchronize()
                ms_per_img = (time.perf_counter() - t0) * 1000.0 / (iters * bs)
            res.setdefault(name, {"o2m_params": params})[f"bs{bs}_ms_per_img"] = round(ms_per_img, 4)
            print(f"    {name:<34}: params={params:,} | {ms_per_img:.4f} ms/img")
    dump(STAGE / "diagnostics/v2_latency_benchmark.json", res)
    return res


if __name__ == "__main__":
    run_qdelta_counterfactual()
    run_tal_assignment_audit()
    run_latency_benchmark()
    print("\nALL 3 DIAGNOSTICS COMPLETE!")
