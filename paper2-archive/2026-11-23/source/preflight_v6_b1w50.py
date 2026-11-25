"""8-Gate GPU+CPU pre-flight verification for V6.1 factorial arms: b1_w50 (B1 + #5) and p2a_w50 (CLS-P2A + #5).

Gates enforced:
  1) Prepare & parent identity: b1_w50 exact state_sha match with records/b1/initial.pt (3,246,130 train / 3,067,810 O2M);
     p2a_w50 exact HistCLS_SchemeA_P2 inheritance (3,208,050); stride-32 alignment at 640 (256/384).
  2) Synthetic image-box alignment: 1024x1024 bright rectangles (upper, lower narrow, straddling y=512) warped to 640
     match warp_bboxes_xywh_np within <= 1.0 px; inverse-y roundtrip < 1e-12.
  3) close_mosaic persistence & no double-warp buffer check on Warp50YOLODataset (6000 imgs / 15619 boxes, cache=None).
  4) CPU warp_image vs cv2.resize latency benchmark (100 iters on real 1024x1024 SVRDD image).
  5) Exact 0.0 pre-NMS raw tensor parity (boxes & scores) on b1 best.pt between YOLODataset and 320/320 Warp50YOLODataset.
  6) 8-img 256/384 smoke on b1 best.pt at 640 & 768 + per-epoch InverseWarpCOCOValidator check (before Gate 7 so val.cache ends clean).
  7) 1,000-img 320/320 identity-warp Val parity gate on CLS-P2A best.pt (<= 0.005 pp) and b1 best.pt (<= 0.015 pp; AR100/APl <= 0.005 pp).
  8) 4-step real warped Mosaic AMP+MuSGD smoke test on b1_w50 (all 7 shallow_side weights update > 0) + post-update FP32 fuse parity.
"""
import copy
import json
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

STAGE = Path(__file__).resolve().parent
sys.path[:0] = [str(STAGE)]
import run_v2
import candidates_v6
import run_v6
from ultralytics.cfg import get_cfg
from ultralytics.data.build import build_dataloader
from ultralytics.data.dataset import YOLODataset
from ultralytics.engine.trainer import BaseTrainer


def check_synthetic_alignment() -> dict:
    im = np.zeros((1024, 1024, 3), dtype=np.uint8)
    rects = [
        ("upper", 50, 100, 200, 180),
        ("lower_narrow", 300, 600, 450, 608),
        ("straddle_horizon", 550, 480, 750, 560),
    ]
    boxes_xywh = []
    for _, x1, y1, x2, y2 in rects:
        im[y1:y2, x1:x2, :] = 255
        boxes_xywh.append([((x1 + x2) * 0.5) / 1024.0, ((y1 + y2) * 0.5) / 1024.0, (x2 - x1) / 1024.0, (y2 - y1) / 1024.0])
    boxes_np = np.array(boxes_xywh, dtype=np.float32)

    warped = candidates_v6.warp_image(im, 640, split_y=0.5, top_frac=0.4)
    assert warped.shape == (640, 640, 3)
    w_boxes = candidates_v6.warp_bboxes_xywh_np(boxes_np, split_y=0.5, top_frac=0.4)

    details = {}
    max_err = 0.0
    for idx, (name, x1, _y1, x2, _y2) in enumerate(rects):
        cx1 = int(round(x1 * 640.0 / 1024.0)) + 2
        cx2 = int(round(x2 * 640.0 / 1024.0)) - 2
        col = warped[:, cx1:cx2, 0].mean(axis=1)
        ys = np.flatnonzero(col > 127.0)
        obs_y1 = float(ys[0])
        obs_y2 = float(ys[-1] + 1)
        exp_y1 = float((w_boxes[idx, 1] - w_boxes[idx, 3] * 0.5) * 640.0)
        exp_y2 = float((w_boxes[idx, 1] + w_boxes[idx, 3] * 0.5) * 640.0)
        err = max(abs(obs_y1 - exp_y1), abs(obs_y2 - exp_y2))
        max_err = max(max_err, err)
        assert err <= 1.0, (name, obs_y1, obs_y2, exp_y1, exp_y2, err)
        details[name] = dict(obs=[obs_y1, obs_y2], exp=[round(exp_y1, 3), round(exp_y2, 3)], err_px=round(err, 4))

    y_pts = np.array([0.0, 100 / 1024, 480 / 1024, 0.5, 560 / 1024, 600 / 1024, 608 / 1024, 1.0], dtype=np.float64)
    rt_err = float(np.max(np.abs(candidates_v6.warp_y_inverse_np(candidates_v6.warp_y_forward_np(y_pts)) - y_pts)))
    assert rt_err < 1e-12
    print(f"  [Gate 2 Synthetic Align OK] max_err={max_err:.3f} px, roundtrip={rt_err:.2e}")
    return dict(max_err_px=max_err, roundtrip_err=rt_err, rects=details)


def run_preflight() -> dict:
    print("\n================ PRE-FLIGHT V6.1 (8 GATES): b1_w50 & p2a_w50 ================")

    # Gate 1: Prepare & exact parent identities + stride-32 breakpoint alignment
    c_b1w50 = run_v6.prepare("b1_w50")
    c_p2aw50 = run_v6.prepare("p2a_w50")
    assert c_b1w50["params"] == 3246130 and c_p2aw50["params"] == 3208050
    b1_contract = json.loads((run_v2.proof("b1") / "contract.json").read_text())
    p2a_contract = json.loads((run_v2.proof("HistCLS_SchemeA_P2") / "contract.json").read_text())
    assert c_b1w50["state_sha"] == b1_contract["state_sha"]
    assert c_p2aw50["state_sha"] == p2a_contract["state_sha"]
    b1w50_checks = json.loads((run_v2.proof("b1_w50") / "candidate_checks.json").read_text())["checks"]
    assert b1w50_checks["matches_b1_initial_exact"] is True and b1w50_checks["deployed_o2m_params"] == 3067810

    top_h, bot_h, eff_top = candidates_v6.effective_warp_frac(640, 0.5, 256.0 / 640.0)
    assert (top_h, bot_h) == (256, 384) and top_h % 32 == 0 and bot_h % 32 == 0
    print(f"  [Gate 1 Prepare & Align OK] b1_w50=3,246,130, p2a_w50=3,208,050, split=({top_h}, {bot_h})")

    # Gate 2: Synthetic image-box alignment
    gate2 = check_synthetic_alignment()

    # Gate 3: close_mosaic persistence & zero-double-warp buffer test
    data = yaml.safe_load(run_v2.DATA_YAML.read_text())
    cfg_tr = get_cfg(overrides=dict(task="detect", data=str(run_v2.DATA_YAML), imgsz=640, batch=8, rect=False, mosaic=1.0, flipud=0.0, fliplr=0.5, workers=0, seed=42))
    ds_tr = candidates_v6.Warp50YOLODataset(
        img_path=str(run_v2.REF / "data/train.txt"),
        imgsz=640,
        batch_size=8,
        augment=True,
        hyp=cfg_tr,
        rect=False,
        cache=None,
        single_cls=False,
        stride=32,
        pad=0.0,
        task="detect",
        data=data,
        warp_split_y=0.5,
        warp_top_frac=0.4,
    )
    assert ds_tr.cache is None and len(ds_tr) == 6000 and sum(len(r["cls"]) for r in ds_tr.labels) == 15619
    raw_im0 = cv2.imread(ds_tr.im_files[0])
    assert raw_im0.shape == (1024, 1024, 3)
    expected_w0 = candidates_v6.warp_image(raw_im0, 640, 0.5, 0.4)
    uniform_0 = cv2.resize(raw_im0, (640, 640), interpolation=cv2.INTER_LINEAR)
    assert int(np.abs(expected_w0.astype(np.int32) - uniform_0.astype(np.int32)).max()) > 0

    lb1 = ds_tr.get_image_and_label(0)
    lb2 = ds_tr.get_image_and_label(0)  # buffered path: must NOT double-warp!
    assert int(np.abs(lb1["img"].astype(np.int32) - expected_w0.astype(np.int32)).max()) == 0
    assert int(np.abs(lb2["img"].astype(np.int32) - expected_w0.astype(np.int32)).max()) == 0
    assert np.allclose(lb1["instances"].bboxes, lb2["instances"].bboxes, atol=1e-7)

    ds_tr.close_mosaic(copy.copy(cfg_tr))
    ds_tr.ims[0] = None
    ds_tr.im_hw0[0] = None
    ds_tr.im_hw[0] = None
    ds_tr.buffer.clear()
    lb_cm = ds_tr.get_image_and_label(0)
    assert int(np.abs(lb_cm["img"].astype(np.int32) - expected_w0.astype(np.int32)).max()) == 0
    sample_cm = ds_tr[0]
    assert tuple(sample_cm["img"].shape) == (3, 640, 640)
    print("  [Gate 3 No-Double-Warp & close_mosaic Persistence OK] diff=0 pre/post close_mosaic")

    # Restore augment=True mosaic=1.0 dataset for Gate 8
    cfg_tr = get_cfg(overrides=dict(task="detect", data=str(run_v2.DATA_YAML), imgsz=640, batch=8, rect=False, mosaic=1.0, flipud=0.0, fliplr=0.5, workers=0, seed=42))
    ds_tr = candidates_v6.Warp50YOLODataset(
        img_path=str(run_v2.REF / "data/train.txt"),
        imgsz=640,
        batch_size=8,
        augment=True,
        hyp=cfg_tr,
        rect=False,
        cache=None,
        single_cls=False,
        stride=32,
        pad=0.0,
        task="detect",
        data=data,
        warp_split_y=0.5,
        warp_top_frac=0.4,
    )

    # Gate 4: CPU Warp latency vs uniform resize (100 iters on real 1024x1024 SVRDD image)
    for _ in range(5):
        _ = candidates_v6.warp_image(raw_im0, 640, 0.5, 0.4)
        _ = cv2.resize(raw_im0, (640, 640), interpolation=cv2.INTER_LINEAR)
    t0 = time.perf_counter()
    for _ in range(100):
        _ = candidates_v6.warp_image(raw_im0, 640, 0.5, 0.4)
    warp_ms = (time.perf_counter() - t0) * 1000.0 / 100.0
    t0 = time.perf_counter()
    for _ in range(100):
        _ = cv2.resize(raw_im0, (640, 640), interpolation=cv2.INTER_LINEAR)
    resize_ms = (time.perf_counter() - t0) * 1000.0 / 100.0
    print(f"  [Gate 4 CPU Latency] uniform_resize={resize_ms:.3f} ms/img, warp50={warp_ms:.3f} ms/img (delta={warp_ms - resize_ms:+.3f} ms)")

    b1_best = STAGE / "runs/b1/weights/best.pt"
    p2a_best = STAGE / "runs/HistCLS_SchemeA_P2/weights/best.pt"
    assert b1_best.is_file() and p2a_best.is_file()

    # Gate 5 & Gate 6: Pre-NMS exact 0.0 raw parity + 8-img 256/384 640/768 & InverseWarpCOCOValidator smoke (BEFORE 1,000-img scan!)
    smoke_8img = {}
    with tempfile.TemporaryDirectory(prefix="v61_smoke_") as td:
        td_p = Path(td)
        sub_txt = td_p / "val8.txt"
        val_lines = (run_v2.REF / "data/val.txt").read_text().splitlines()
        sub_txt.write_text("\n".join(val_lines[:8]) + "\n")
        sub_yaml = td_p / "val8.yaml"
        cfg_y = yaml.safe_load(run_v2.DATA_YAML.read_text())
        cfg_y["val"] = str(sub_txt)
        sub_yaml.write_text(yaml.safe_dump(cfg_y, sort_keys=False))

        cfg_v4 = get_cfg(overrides=dict(task="detect", data=str(sub_yaml), imgsz=640, batch=4, rect=True, split="val", workers=0))
        d_sub = yaml.safe_load(sub_yaml.read_text())
        ds_std = YOLODataset(img_path=str(sub_txt), imgsz=640, batch_size=4, augment=False, hyp=cfg_v4, rect=True, cache=None, single_cls=False, stride=32, pad=0.5, task="detect", data=d_sub)
        ds_idw = candidates_v6.Warp50YOLODataset(img_path=str(sub_txt), imgsz=640, batch_size=4, augment=False, hyp=cfg_v4, rect=True, cache=None, single_cls=False, stride=32, pad=0.5, task="detect", data=d_sub, warp_split_y=0.5, warp_top_frac=0.5)
        b_std = YOLODataset.collate_fn([ds_std[i] for i in range(4)])
        b_idw = YOLODataset.collate_fn([ds_idw[i] for i in range(4)])
        img_diff = int((b_std["img"].to(torch.int32) - b_idw["img"].to(torch.int32)).abs().max())
        assert img_diff == 0, f"Identity dataset image diff: {img_diff}"

        prev_mm, prev_cn = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        try:
            fm = candidates_v6.compact_o2m(run_v2.reference.load_model(b1_best)).cuda().eval()
            with torch.no_grad():
                x_s = b_std["img"].cuda().float() / 255.0
                x_i = b_idw["img"].cuda().float() / 255.0
                ys, _ = fm(x_s)
                yi, _ = fm(x_i)
            raw_box_diff = float((ys[:, :4] - yi[:, :4]).abs().max())
            raw_score_diff = float((ys[:, 4:] - yi[:, 4:]).abs().max())
        finally:
            torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = prev_mm, prev_cn
        assert raw_box_diff == 0.0 and raw_score_diff == 0.0, (raw_box_diff, raw_score_diff)
        print(f"  [Gate 5 Pre-NMS Raw Tensor Parity OK (b1 best.pt)] img_diff={img_diff}, box_diff={raw_box_diff}, score_diff={raw_score_diff}")

        for imgsz in (640, 768):
            val8 = run_v6.InverseWarpTaskValidator(
                args=dict(task="detect", data=str(sub_yaml), imgsz=imgsz, batch=8, workers=0, device="0", conf=0.001, iou=0.7, max_det=300, rect=True, save_json=True, plots=False, verbose=False, half=False, split="val"),
                save_dir=td_p / f"out_{imgsz}",
            )
            m8 = val8(model=run_v2.reference.load_model(b1_best))
            assert val8.seen == 8 and len(val8.jdict) > 0
            smoke_8img[str(imgsz)] = dict(
                preds=len(val8.jdict),
                native_mAP50_pct=round(float(m8["metrics/mAP50(B)"]) * 100.0, 3),
                native_mAP50_95_pct=round(float(m8["metrics/mAP50-95(B)"]) * 100.0, 3),
            )

        val8_coco = run_v6.InverseWarpCOCOValidator(
            args=dict(task="detect", data=str(sub_yaml), imgsz=640, batch=8, workers=0, device="0", conf=0.001, iou=0.7, max_det=300, rect=True, save_json=True, plots=False, verbose=False, half=False, split="val"),
            save_dir=td_p / "out_coco_640",
        )
        m8_coco = val8_coco(model=run_v2.reference.load_model(b1_best))
        assert len(val8_coco.evaluated_ids) == 8 and 0.0 <= val8_coco.coco_main["AP50_95"] <= 1.0
        smoke_8img["coco_selector_640"] = {k: round(float(v) * 100.0, 3) for k, v in val8_coco.coco_main.items()}
        assert "COCO/AP50_95" in m8_coco
        print(f"  [Gate 6 8-img 256/384 + InverseWarpCOCOValidator OK] {json.dumps(smoke_8img)}")

    # Gate 7: 1,000-image 320/320 identity-warp Val parity on BOTH CLS-P2A best.pt and b1 best.pt (leaves 1000-img val.cache clean)
    parity_1000 = {}
    for arm_name, ck_path, tol_main in (
        ("CLS_P2A", p2a_best, 0.005),
        ("b1", b1_best, 0.015),
    ):
        rec_dir = "HistCLS_SchemeA_P2" if arm_name == "CLS_P2A" else "b1"
        off = json.loads((run_v2.proof(rec_dir) / "evaluation/summary.json").read_text())["best"]
        with tempfile.TemporaryDirectory(prefix=f"v61_id_{arm_name}_") as td:
            id_res = run_v6.evaluate_warp_at_imgsz(ck_path, Path(td) / "eval", imgsz=640, batch=32, split_y=0.5, top_frac=0.5)
        diffs_pp = {
            k: round(abs(id_res["main"][k] - off["main"][k]) * 100.0, 5)
            for k in ("AP50_95", "AP50", "AP75", "AR100", "AP_small", "AP_medium", "AP_large")
        }
        nat_keys = ("metrics/mAP50(B)", "metrics/mAP50-95(B)") + (("metrics/AP75(B)",) if "metrics/AP75(B)" in off["native_metrics"] else ())
        nat_diff_pp = round(max(abs(id_res["native_metrics"][k] - off["native_metrics"][k]) * 100.0 for k in nat_keys), 5)
        max_pp = max(max(diffs_pp.values()), nat_diff_pp)
        assert max_pp <= tol_main, (arm_name, max_pp, diffs_pp, nat_diff_pp)
        if arm_name == "b1":
            assert diffs_pp["AR100"] <= 0.005 and diffs_pp["AP_large"] <= 0.005, diffs_pp
        parity_1000[arm_name] = dict(max_diff_pp=max_pp, diffs_pp=diffs_pp, native_diff_pp=nat_diff_pp)
        print(f"  [Gate 7 Identity 320/320 Parity OK ({arm_name})] max_diff={max_pp:.5f} pp | diffs={json.dumps(diffs_pp)}")

    # Gate 8: 4-step real warped Mosaic AMP+MuSGD + post-update FP32 fuse parity on b1_w50
    model = run_v2.reference.load_model(c_b1w50["initial"]).cuda()
    model.train()
    crit = model.init_criterion()
    assert all(branch.assigner.stride == [8.0, 16.0, 32.0] and branch.reg_max == 1 for branch in (crit.one2many, crit.one2one))
    model.criterion = crit

    dummy = type("Dummy", (), {"args": model.args, "build_optimizer": BaseTrainer.build_optimizer})()
    opt = dummy.build_optimizer(model, name="MuSGD", lr=0.001, momentum=0.9, decay=0.0005, iterations=10000)
    assert {id(p) for p in model.parameters() if p.requires_grad} == {id(p) for g in opt.param_groups for p in g["params"]}
    side_names = [name for name, _ in model.named_parameters() if "shallow_side." in name]
    assert len(side_names) == 12
    init_snap = {name: p.detach().clone() for name, p in model.named_parameters() if name in side_names}

    dl = build_dataloader(ds_tr, batch=8, workers=0, shuffle=False, rank=-1)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    it_dl = iter(dl)
    succ = 0
    for step in range(1, 16):
        batch = next(it_dl)
        batch["img"] = batch["img"].cuda(non_blocking=True).float() / 255.0
        for k in ("batch_idx", "cls", "bboxes"):
            batch[k] = batch[k].cuda(non_blocking=True)
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=True):
            preds = model(batch["img"])
            loss_vec, loss_items = crit(preds, batch)
            loss = loss_vec.sum()
        assert torch.isfinite(loss).all() and torch.isfinite(loss_items).all()
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        before = scaler.get_scale()
        scaler.step(opt)
        scaler.update()
        after = scaler.get_scale()
        if after >= before:
            succ += 1
        print(f"  b1_w50 real-batch AMP+MuSGD step {step}: loss={float(loss):.4f}, scale={before:.0f}->{after:.0f}")
        if succ >= 4:
            break
    assert succ >= 4
    for name, p in model.named_parameters():
        assert torch.isfinite(p).all(), f"Non-finite parameter: {name}"

    side_deltas = {}
    for key in ("reduce.conv.weight", "deconv.vc.weight", "deconv.cdc.weight", "deconv.adc.weight", "deconv.hdc.weight", "deconv.vdc.weight", "proj.weight"):
        full_name = f"model.2.shallow_side.{key}"
        delta = float((dict(model.named_parameters())[full_name].detach() - init_snap[full_name]).abs().max())
        assert delta > 0.0, f"{full_name} did NOT update on real warped Mosaic batch!"
        side_deltas[key] = delta

    prev_mm, prev_cn = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    fuse_err = {}
    try:
        from rebuilt_model import as_o2m

        model.eval()
        u_o2m = as_o2m(copy.deepcopy(model)).cuda().eval()
        f_o2m = candidates_v6.compact_o2m(copy.deepcopy(model)).cuda().eval()
        assert sum(isinstance(m, torch.nn.BatchNorm2d) for m in f_o2m.model[2].shallow_side.modules()) == 0
        assert sum(isinstance(m, torch.nn.BatchNorm2d) for m in f_o2m.modules()) == 20
        for sh in ((2, 3, 480, 640), (1, 3, 768, 768)):
            with torch.no_grad():
                xv = torch.rand(*sh, device="cuda", generator=torch.Generator(device="cuda").manual_seed(9))
                yu = u_o2m(xv)[0]
                yf = f_o2m(xv)[0]
            be = float((yu[:, :4] - yf[:, :4]).abs().max())
            se = float((yu[:, 4:] - yf[:, 4:]).abs().max())
            assert be < 1e-2 and se < 1e-4, (sh, be, se)
            fuse_err[f"{sh[2]}x{sh[3]}"] = dict(box_err_px=be, score_err=se)
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = prev_mm, prev_cn

    print("  [Gate 8 AMP+MuSGD + Post-Update Fuse Parity OK] 7/7 side weights updated, fuse_err=", json.dumps(fuse_err))

    receipt = dict(
        b1_w50=dict(params=c_b1w50["params"], deployed_o2m_params=b1w50_checks["deployed_o2m_params"], state_sha=c_b1w50["state_sha"], matches_b1_initial_exact=True),
        p2a_w50=dict(params=c_p2aw50["params"], state_sha=c_p2aw50["state_sha"], matches_p2a_initial_exact=True),
        breakpoint_640=dict(top_h=top_h, bot_h=bot_h, stride32_aligned=True, eff_top_frac=eff_top),
        gate2_synthetic_align=gate2,
        gate3_no_double_warp_and_close_mosaic_ok=True,
        gate4_cpu_latency_ms=dict(uniform_resize=round(resize_ms, 4), warp50=round(warp_ms, 4), delta=round(warp_ms - resize_ms, 4)),
        gate5_pre_nms_raw_parity=dict(img_diff=img_diff, box_diff=raw_box_diff, score_diff=raw_score_diff),
        gate6_smoke_8img_256_384=smoke_8img,
        gate7_identity_320_parity_1000=parity_1000,
        gate8_amp_musgd_steps=succ,
        gate8_side_weight_max_deltas=side_deltas,
        gate8_post_update_fuse_parity=fuse_err,
    )
    run_v6.dump(STAGE / "records/preflight_v6_b1w50.json", receipt)
    print("\nALL 8 V6.1 PRE-FLIGHT GATES PASSED. Saved records/preflight_v6_b1w50.json")
    return receipt


if __name__ == "__main__":
    run_preflight()
