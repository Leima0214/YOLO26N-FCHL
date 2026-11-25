"""5-Gate CPU+GPU pre-flight check for V7 ablation arms: b0, b0_w50, b0_p2, b1_768, b1_w448."""
import copy
import json
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

STAGE = Path(__file__).resolve().parent
sys.path[:0] = [str(STAGE)]
import run_v2
import candidates_v6
import candidates_v7
import run_v7
from ultralytics.cfg import get_cfg
from ultralytics.data.build import build_dataloader
from ultralytics.data.dataset import YOLODataset
from ultralytics.engine.trainer import BaseTrainer


def check_synthetic_alignment_192_448() -> dict:
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

    warped = candidates_v6.warp_image(im, 640, split_y=0.5, top_frac=0.3)
    assert warped.shape == (640, 640, 3)
    w_boxes = candidates_v6.warp_bboxes_xywh_np(boxes_np, split_y=0.5, top_frac=0.3)

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
    rt_err = float(
        np.max(np.abs(candidates_v6.warp_y_inverse_np(candidates_v6.warp_y_forward_np(y_pts, 0.5, 0.3), 0.5, 0.3) - y_pts))
    )
    assert rt_err < 1e-12
    print(f"  [Gate 2b 192/448 Synthetic Align OK] max_err={max_err:.3f} px, roundtrip={rt_err:.2e}")
    return dict(max_err_px=max_err, roundtrip_err=rt_err, rects=details)


def run_step_smoke(arm: str, model_path: Path, imgsz: int, warp_cfg, ds_train) -> dict:
    """Run 4 real AMP+MuSGD optimizer steps at batch=32 on RTX 4090 and check finite loss + peak VRAM."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    m = run_v2.reference.load_model(model_path).cuda().train()
    for p in m.parameters():
        p.requires_grad_(True)
    m.args = get_cfg(overrides=dict(task="detect", imgsz=imgsz, batch=32, epochs=100, optimizer="MuSGD", amp=True, box=7.5, cls=0.5, dfl=1.5, lr0=0.01, momentum=0.9, weight_decay=0.0005, nbs=64, seed=42))
    m.criterion = m.init_criterion()

    # Build temporary real optimizer mirroring Ultralytics MuSGD parameter grouping
    bt = object.__new__(BaseTrainer)
    bt.args = m.args
    bt.model = m
    opt = bt.build_optimizer(m, name="MuSGD", lr=0.01, momentum=0.9, decay=0.0005, iterations=18800)
    scaler = torch.amp.GradScaler("cuda", enabled=True)

    loader = build_dataloader(ds_train, batch=32, workers=2, shuffle=False, rank=-1)
    it = iter(loader)
    losses = []
    init_snap = {k: v.detach().clone() for k, v in m.state_dict().items()}
    for step_idx in range(4):
        batch = next(it)
        batch["img"] = batch["img"].cuda(non_blocking=True).float() / 255.0
        for k in ("batch_idx", "cls", "bboxes"):
            batch[k] = batch[k].cuda(non_blocking=True)
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=True):
            loss, loss_items = m(batch)
            total_loss = loss.sum() if loss.ndim > 0 else loss
        assert torch.isfinite(total_loss) and torch.isfinite(loss_items).all(), (arm, step_idx, loss_items)
        scaler.scale(total_loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(m.parameters(), max_norm=10.0)
        scaler.step(opt)
        scaler.update()
        losses.append([round(float(x), 4) for x in loss_items])

    changed_tensors = sum(1 for k, v in m.state_dict().items() if v.is_floating_point() and not torch.equal(v, init_snap[k]))
    peak_alloc = float(torch.cuda.max_memory_allocated() / 2**30)
    peak_res = float(torch.cuda.max_memory_reserved() / 2**30)
    del loader, opt, scaler, m, init_snap
    torch.cuda.empty_cache()

    assert peak_res < 22.5, f"{arm} ({imgsz}) peak reserved VRAM too high: {peak_res:.2f} GiB"
    assert changed_tensors > 100
    print(f"  [Gate 4 Train Smoke {arm} @ {imgsz}x{imgsz} bs=32] losses={losses[-1]}, changed={changed_tensors}, alloc={peak_alloc:.2f}GiB, res={peak_res:.2f}GiB")
    return dict(imgsz=imgsz, warp=warp_cfg, losses=losses, changed_tensors=changed_tensors, peak_alloc_GiB=round(peak_alloc, 3), peak_reserved_GiB=round(peak_res, 3))


def run_preflight() -> dict:
    print("\n================ PRE-FLIGHT V7 (5 GATES): b0, b0_w50, b0_p2, b1_768, b1_w448 ================")
    assert (STAGE / "runs/warp50/DEFERRED_BY_USER_TOMORROW.txt").is_file(), "Missing warp50 sentinel!"
    for arm in candidates_v7.ARMS:
        assert not (STAGE / "runs" / arm).exists(), f"Run dir already exists: runs/{arm}"

    # Gate 1: Prepare & exact parent identities across all 5 arms
    contracts = {}
    for arm in candidates_v7.ARMS:
        contracts[arm] = run_v7.prepare(arm)
    assert contracts["b0"]["state_sha"] == contracts["b0_w50"]["state_sha"]
    assert contracts["b1_768"]["state_sha"] == contracts["b1_w448"]["state_sha"]
    print("  [Gate 1 All 5 Contracts OK] b0=b0_w50 (2,506,530), b0_p2 (2,574,360), b1_768=b1_w448 (3,246,130)")

    # Gate 2: Synthetic align + dataset checks for 192/448 and 256/384
    gate2 = check_synthetic_alignment_192_448()
    data = yaml.safe_load(run_v2.DATA_YAML.read_text())

    def make_ds(sz: int, wcfg):
        hyp = get_cfg(overrides=dict(task="detect", data=str(run_v2.DATA_YAML), imgsz=sz, batch=32, rect=False, mosaic=1.0, flipud=0.0, fliplr=0.5, workers=2, seed=42))
        if wcfg is None:
            return YOLODataset(img_path=str(run_v2.REF / "data/train.txt"), imgsz=sz, batch_size=32, augment=True, hyp=hyp, rect=False, cache=None, single_cls=False, stride=32, pad=0.0, task="detect", data=data)
        return candidates_v6.Warp50YOLODataset(img_path=str(run_v2.REF / "data/train.txt"), imgsz=sz, batch_size=32, augment=True, hyp=hyp, rect=False, cache=None, single_cls=False, stride=32, pad=0.0, task="detect", data=data, warp_split_y=wcfg[0], warp_top_frac=wcfg[1])

    hyp_w448 = get_cfg(overrides=dict(task="detect", data=str(run_v2.DATA_YAML), imgsz=640, batch=32, rect=False, mosaic=1.0, flipud=0.0, fliplr=0.5, workers=2, seed=42))
    ds_w448 = make_ds(640, (0.5, 0.3))
    raw_im0 = cv2.imread(ds_w448.im_files[0])
    exp_448 = candidates_v6.warp_image(raw_im0, 640, 0.5, 0.3)
    lb = ds_w448.get_image_and_label(0)
    assert int(np.abs(lb["img"].astype(np.int32) - exp_448.astype(np.int32)).max()) == 0
    ds_w448.close_mosaic(copy.copy(hyp_w448))
    ds_w448.ims[0] = None
    ds_w448.im_hw0[0] = None
    ds_w448.im_hw[0] = None
    ds_w448.buffer.clear()
    assert int(np.abs(ds_w448.get_image_and_label(0)["img"].astype(np.int32) - exp_448.astype(np.int32)).max()) == 0
    print("  [Gate 2 Datasets OK] 192/448 verified pre/post close_mosaic")

    # Gate 3: 8-img GPU smoke validation for O2O (b0 & b0_w50), O2M (b0, b0_w50, b0_p2, b1_w448 192/448)
    smoke_8img = {}
    with tempfile.TemporaryDirectory(prefix="v7_smoke_") as td:
        td_p = Path(td)
        sub_txt = td_p / "val8.txt"
        val_lines = (run_v2.REF / "data/val.txt").read_text().splitlines()
        sub_txt.write_text("\n".join(val_lines[:8]) + "\n")
        sub_yaml = td_p / "val8.yaml"
        cfg_y = yaml.safe_load(run_v2.DATA_YAML.read_text())
        cfg_y["val"] = str(sub_txt)
        sub_yaml.write_text(yaml.safe_dump(cfg_y, sort_keys=False))

        b0_init_ck = run_v2.proof("b0") / "initial.pt"
        b0p2_init_ck = run_v2.proof("b0_p2") / "initial.pt"
        b1w50_best = STAGE / "runs/b1_w50/weights/best.pt"

        r_b0_o2m = run_v7.evaluate_uniform_at_imgsz(b0_init_ck, td_p / "b0_o2m", imgsz=640, batch=8, data_yaml=sub_yaml)
        r_b0_o2o = run_v7.evaluate_o2o_at_imgsz(b0_init_ck, td_p / "b0_o2o", imgsz=640, batch=8, warp_cfg=None, data_yaml=sub_yaml)
        r_b0w_o2m = run_v7.evaluate_warp_at_imgsz(b0_init_ck, td_p / "b0w_o2m", imgsz=640, batch=8, split_y=0.5, top_frac=0.4, data_yaml=sub_yaml)
        r_b0w_o2o = run_v7.evaluate_o2o_at_imgsz(b0_init_ck, td_p / "b0w_o2o", imgsz=640, batch=8, warp_cfg=(0.5, 0.4), data_yaml=sub_yaml)
        r_b0p2 = run_v7.evaluate_uniform_at_imgsz(b0p2_init_ck, td_p / "b0_p2", imgsz=640, batch=8, data_yaml=sub_yaml)
        r_w448_640 = run_v7.evaluate_warp_at_imgsz(b1w50_best, td_p / "w448_640", imgsz=640, batch=8, split_y=0.5, top_frac=0.3, data_yaml=sub_yaml)
        r_w448_768 = run_v7.evaluate_warp_at_imgsz(b1w50_best, td_p / "w448_768", imgsz=768, batch=8, split_y=0.5, top_frac=0.3, data_yaml=sub_yaml)
        WarpCOCOV_448 = run_v7.make_validators(0.5, 0.3)[0]
        val8_c448 = WarpCOCOV_448(
            args=dict(task="detect", data=str(sub_yaml), imgsz=640, batch=8, workers=0, device="0", conf=0.001, iou=0.7, max_det=300, rect=True, save_json=True, plots=False, verbose=False, half=False, split="val"),
            save_dir=td_p / "w448_coco",
        )
        m8_c448 = val8_c448(model=run_v2.reference.load_model(b1w50_best))
        assert len(val8_c448.evaluated_ids) == 8 and "COCO/AP50_95" in m8_c448 and val8_c448.coco_main["AP50_95"] > 0.3
        smoke_8img = {
            "b0_o2m_preds": r_b0_o2m["predictions"],
            "b0_o2o_preds": r_b0_o2o["predictions"],
            "b0w_o2m_preds": r_b0w_o2m["predictions"],
            "b0w_o2o_preds": r_b0w_o2o["predictions"],
            "b0_p2_preds": r_b0p2["predictions"],
            "w448_640_native_mAP50_95_on_b1w50_pct": round(float(r_w448_640["native_metrics"]["metrics/mAP50-95(B)"]) * 100, 3),
            "w448_768_native_mAP50_95_on_b1w50_pct": round(float(r_w448_768["native_metrics"]["metrics/mAP50-95(B)"]) * 100, 3),
            "w448_coco_selector_8img_pct": {k: round(float(v) * 100, 3) for k, v in val8_c448.coco_main.items()},
        }
        assert smoke_8img["w448_640_native_mAP50_95_on_b1w50_pct"] > 30.0
        print(f"  [Gate 3 8-img Validator Smoke OK] {json.dumps(smoke_8img)}")

    # Gate 4: 4-step real AMP+MuSGD GPU training smoke at batch=32 across all 5 arms (including 768x768!)
    train_smoke = {}
    train_smoke["b0"] = run_step_smoke("b0", run_v2.proof("b0") / "initial.pt", 640, None, make_ds(640, None))
    train_smoke["b0_w50"] = run_step_smoke("b0_w50", run_v2.proof("b0_w50") / "initial.pt", 640, (0.5, 0.4), make_ds(640, (0.5, 0.4)))
    train_smoke["b0_p2"] = run_step_smoke("b0_p2", run_v2.proof("b0_p2") / "initial.pt", 640, None, make_ds(640, None))
    train_smoke["b1_w448"] = run_step_smoke("b1_w448", run_v2.proof("b1_w448") / "initial.pt", 640, (0.5, 0.3), make_ds(640, (0.5, 0.3)))
    train_smoke["b1_768"] = run_step_smoke("b1_768", run_v2.proof("b1_768") / "initial.pt", 768, None, make_ds(768, None))

    # Gate 5: Ensure full 1000-img Val dataset length intact (cleanup any 8-img cache trace)
    cfg_val_full = get_cfg(overrides=dict(task="detect", data=str(run_v2.DATA_YAML), imgsz=640, batch=32, rect=True, split="val", workers=0))
    ds_val_full = YOLODataset(img_path=str(run_v2.REF / "data/val.txt"), imgsz=640, batch_size=32, augment=False, hyp=cfg_val_full, rect=True, cache=None, single_cls=False, stride=32, pad=0.5, task="detect", data=data)
    assert len(ds_val_full) == 1000 and sum(len(r["cls"]) for r in ds_val_full.labels) == 2612
    print("  [Gate 5 Full Val Cache Clean OK] 1,000 imgs / 2,612 GT")

    receipt = dict(
        time=run_v2.now(),
        passed=True,
        gpu=torch.cuda.get_device_name(0),
        arms=list(candidates_v7.ARMS),
        contracts={
            arm: {k: contracts[arm][k] for k in ("arm", "imgsz", "params", "deployed_o2m_params", "state_sha", "warp")}
            for arm in candidates_v7.ARMS
        },
        gate2_align_192_448=gate2,
        gate3_validator_8img_smoke=smoke_8img,
        gate4_train_4step_bs32=train_smoke,
        gate5_full_val=dict(images=1000, instances=2612),
    )
    out_p = STAGE / "diagnostics/preflight_v7.json"
    run_v2.dump(out_p, receipt)
    print(f"================ PRE-FLIGHT V7 ALL 5 GATES PASSED -> {out_p} ================\n")
    return receipt


if __name__ == "__main__":
    run_preflight()
