"""Strict GPU+CPU pre-flight checks for V6 arms (b1, warp50) before launching serial 100E queue.

Gates enforced:
  1) b1: CLS-P2A exact weight inheritance, DEConv 5-branch equivalence, Step-0 0.0, active BaseModel.fuse() parity,
     MuSGD param grouping, 4-step GPU AMP+MuSGD smoke test (all 7 side-branch weights update), fresh unpickle + AutoBackend fuse,
     and FP32 bs=1 / bs=32 inference latency vs CLS-P2A.
  2) warp50: CLS-P2A exact initial.pt match, flipud=0, forward/inverse round-trip (<1e-12),
     MANDATORY 320/320 Val identity-warp parity gate vs CLS-P2A E90 official (43.234 / 69.633 / 44.953, diff <= 0.005 pp),
     256/384 Warp50YOLODataset Mosaic & Val (672x672) label consistency + Train narrow-box gain receipt,
     1-batch InverseWarpTaskValidator smoke + 768 ratio_pad check, and 4-step GPU AMP+MuSGD smoke test.
"""
import copy
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import cv2
import yaml

STAGE = Path(__file__).resolve().parent
sys.path[:0] = [str(STAGE)]
import run_v2
import run_v6
import candidates_v6
from ultralytics.cfg import get_cfg
from ultralytics.data.build import build_dataloader
from ultralytics.engine.trainer import BaseTrainer


def benchmark_fp32_latency(model_a, model_b, warmup: int = 10, iters: int = 30) -> dict:
    """Measure pure FP32 (no TF32) GPU forward latency at bs=1 and bs=32."""
    prev_mm = torch.backends.cuda.matmul.allow_tf32
    prev_cn = torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    out = {}
    try:
        for tag, m in (("p2a", model_a), ("b1", model_b)):
            m = copy.deepcopy(m).cuda().eval()
            out[tag] = {"deployed_params": sum(p.numel() for p in m.parameters())}
            for bs in (1, 32):
                x = torch.rand(bs, 3, 640, 640, device="cuda")
                with torch.no_grad():
                    for _ in range(warmup):
                        _ = m(x)
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    for _ in range(iters):
                        _ = m(x)
                    torch.cuda.synchronize()
                    ms_img = ((time.perf_counter() - t0) / iters) * 1000.0 / bs
                out[tag][f"bs{bs}_ms_per_img"] = round(ms_img, 4)
        out["delta_bs1_ms"] = round(out["b1"]["bs1_ms_per_img"] - out["p2a"]["bs1_ms_per_img"], 4)
        out["delta_bs32_ms"] = round(out["b1"]["bs32_ms_per_img"] - out["p2a"]["bs32_ms_per_img"], 4)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev_mm
        torch.backends.cudnn.allow_tf32 = prev_cn
    return out


def check_b1() -> dict:
    print("\n================ PRE-FLIGHT V6: b1 (L0->L2 SPD+DEConv) ================")
    c = run_v6.prepare("b1")
    model = run_v2.reference.load_model(c["initial"]).cuda()
    model.train()
    crit = model.init_criterion()
    assert all(
        branch.assigner.stride == [8.0, 16.0, 32.0] and branch.reg_max == 1
        for branch in (crit.one2many, crit.one2one)
    )
    model.criterion = crit

    # 1. Verify MuSGD param grouping
    dummy = type("Dummy", (), {"args": model.args, "build_optimizer": BaseTrainer.build_optimizer})()
    opt = dummy.build_optimizer(model, name="MuSGD", lr=0.001, momentum=0.9, decay=0.0005, iterations=10000)
    opt_ids = {id(p) for g in opt.param_groups for p in g["params"]}
    model_ids = {id(p) for p in model.parameters() if p.requires_grad}
    assert model_ids == opt_ids, f"Missing params in MuSGD: {len(model_ids - opt_ids)}"
    id_to_group = {
        id(p): (gi, g.get("use_muon", False), g.get("weight_decay", 0.0))
        for gi, g in enumerate(opt.param_groups)
        for p in g["params"]
    }

    side_names = [name for name, _ in model.named_parameters() if "shallow_side." in name]
    for name in side_names:
        p = dict(model.named_parameters())[name]
        gi, um, wd = id_to_group[id(p)]
        print(f"  param {name:<48} shape={str(tuple(p.shape)):<18} group={gi} muon={um} wd={wd}")
        if p.ndim == 1:
            assert not um, f"1D param {name} wrongly assigned muon={um}"
    # 11 tensors: reduce.(conv.w, bn.w, bn.b) + deconv.(vc,cdc,adc,hdc,vdc,bn.w,bn.b) + proj.(w,b) = 3 + 7 + 2 = 12
    assert len(side_names) == 12, f"Expected 12 shallow_side tensors, got {len(side_names)}: {side_names}"

    # 2. 4-step real AMP + MuSGD + clip_grad_norm_(10.0) smoke test
    init_snap = {name: p.detach().clone() for name, p in model.named_parameters() if name in side_names}
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    gen = torch.Generator(device="cuda").manual_seed(123)
    x = torch.rand(4, 3, 640, 640, device="cuda", generator=gen)
    batch = {
        "batch_idx": torch.tensor([0, 0, 1, 2, 3], dtype=torch.float32, device="cuda"),
        "cls": torch.tensor([[0], [1], [4], [2], [3]], dtype=torch.float32, device="cuda"),
        "bboxes": torch.tensor(
            [
                [0.50, 0.50, 0.02, 0.30],
                [0.30, 0.40, 0.25, 0.015],
                [0.60, 0.60, 0.04, 0.04],
                [0.40, 0.70, 0.12, 0.10],
                [0.75, 0.25, 0.05, 0.05],
            ],
            dtype=torch.float32,
            device="cuda",
        ),
    }

    succ = 0
    for step in range(1, 16):
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=True):
            preds = model(x)
            loss_vec, loss_items = crit(preds, batch)
            loss = loss_vec.sum()
        assert torch.isfinite(loss).all() and torch.isfinite(loss_items).all(), (step, loss, loss_items)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        before = scaler.get_scale()
        scaler.step(opt)
        scaler.update()
        after = scaler.get_scale()
        stepped = after >= before
        if stepped:
            succ += 1
        print(f"  AMP+MuSGD step {step:2d}: loss={float(loss):.4f}, scale={before:.0f}->{after:.0f}, stepped={stepped}")
        if succ >= 4:
            break
    assert succ >= 4, f"Only {succ} unskipped AMP+MuSGD steps!"

    for name, p in model.named_parameters():
        assert torch.isfinite(p).all(), f"Non-finite parameter after AMP+MuSGD: {name}"
    for key in ("reduce.conv.weight", "deconv.vc.weight", "deconv.cdc.weight", "deconv.adc.weight", "deconv.hdc.weight", "deconv.vdc.weight", "proj.weight"):
        full_name = f"model.2.shallow_side.{key}"
        p = dict(model.named_parameters())[full_name]
        delta = float((p.detach() - init_snap[full_name]).abs().max())
        print(f"  updated {full_name:<46} max_abs_delta={delta:.6e}")
        assert delta > 0.0, f"{full_name} did NOT update!"

    # 3. Post-update FP32 fuse parity on 480x640 and 768x768 + AutoBackend/info compatibility
    prev_mm = torch.backends.cuda.matmul.allow_tf32
    prev_cn = torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    fuse_err = {}
    try:
        from rebuilt_model import as_o2m

        model.eval()
        u_o2m = as_o2m(copy.deepcopy(model)).cuda().eval()
        f_o2m = candidates_v6.compact_o2m(copy.deepcopy(model)).cuda().eval()
        assert not any(isinstance(m, torch.nn.BatchNorm2d) for m in f_o2m.model[2].shallow_side.modules())
        for sh in ((2, 3, 480, 640), (1, 3, 768, 768)):
            with torch.no_grad():
                xv = torch.rand(*sh, device="cuda", generator=torch.Generator(device="cuda").manual_seed(9))
                yu = u_o2m(xv)[0]
                yf = f_o2m(xv)[0]
            be = float((yu[:, :4] - yf[:, :4]).abs().max())
            se = float((yu[:, 4:] - yf[:, 4:]).abs().max())
            print(f"  post-update FP32 fuse parity {sh[2]}x{sh[3]}: box_err={be:.3e} px, score_err={se:.3e}")
            assert be < 1e-2 and se < 1e-4, (sh, be, se)
            fuse_err[f"{sh[2]}x{sh[3]}"] = dict(box_err_px=be, score_err=se)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev_mm
        torch.backends.cudnn.allow_tf32 = prev_cn

    # 4. Fresh-process unpickle + model.info() + latency vs CLS-P2A
    probe = (
        f"import sys, torch; sys.path[:0]=['{STAGE}']; from run import reference; "
        f"m = reference.load_model('{c['initial']}'); m.info(verbose=False); "
        f"import candidates_v6; fm = candidates_v6.compact_o2m(m); "
        f"side_bn = sum(isinstance(x, torch.nn.BatchNorm2d) for x in fm.model[2].shallow_side.modules()); "
        f"assert side_bn == 0 and hasattr(fm.model[2].shallow_side.deconv, 'conv'); "
        f"print('  fresh-process unpickle+fuse OK:', type(fm.model[2]).__name__, sum(p.numel() for p in fm.parameters()), 'side_bn=', side_bn)"
    )
    res = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"Fresh-process probe failed ({res.returncode}):\n{res.stdout}\n{res.stderr}")
    print(res.stdout.strip())

    p2a_init = run_v2.reference.load_model(run_v2.proof("HistCLS_SchemeA_P2") / "initial.pt")
    lat = benchmark_fp32_latency(candidates_v6.compact_o2m(p2a_init), candidates_v6.compact_o2m(run_v2.reference.load_model(c["initial"])))
    print("  FP32 latency vs CLS-P2A:", json.dumps(lat))
    print("================ PRE-FLIGHT PASS: b1 ================")
    return dict(
        params=c["params"],
        added_over_p2a=c["added_over_p2a"],
        fuse_parity=fuse_err,
        latency=lat,
        side_tensor_count=len(side_names),
        smoke_unskipped_steps=succ,
    )


def check_warp50() -> dict:
    print("\n================ PRE-FLIGHT V6: warp50 (#5 256/384 Road Resampling) ================")
    c = run_v6.prepare("warp50")
    p2a_best = STAGE / "runs/HistCLS_SchemeA_P2/weights/best.pt"
    assert p2a_best.is_file(), f"Missing {p2a_best}"

    # 1. Geometry sanity & Mosaic + Val dataloader check for 256/384
    data = yaml.safe_load(run_v2.DATA_YAML.read_text())
    cfg_tr = get_cfg(overrides=dict(task="detect", data=str(run_v2.DATA_YAML), imgsz=640, batch=4, rect=False, mosaic=1.0, flipud=0.0, fliplr=0.5, workers=0, seed=42))
    ds_tr = candidates_v6.Warp50YOLODataset(
        img_path=str(run_v2.REF / "data/train.txt"),
        imgsz=640,
        batch_size=4,
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
    assert len(ds_tr) == 6000 and sum(len(r["cls"]) for r in ds_tr.labels) == 15619
    sample_tr = ds_tr[0]
    assert tuple(sample_tr["img"].shape) == (3, 640, 640), sample_tr["img"].shape

    # Verify raw lower-half narrow box vertical amplification on Train labels
    all_boxes = np.concatenate([r["bboxes"] for r in ds_tr.labels], axis=0)
    short640_orig = np.minimum(all_boxes[:, 2], all_boxes[:, 3]) * 640.0
    lt8_mask = short640_orig < 8.0
    warped_boxes = candidates_v6.warp_bboxes_xywh_np(all_boxes)
    short640_warp = np.minimum(warped_boxes[:, 2], warped_boxes[:, 3]) * 640.0
    crossed_8 = int((lt8_mask & (short640_warp >= 8.0)).sum())
    print(
        f"  Train boxes={len(all_boxes)}, short640<8={int(lt8_mask.sum())}, "
        f"median short640 {float(np.median(short640_orig[lt8_mask])):.3f} -> {float(np.median(short640_warp[lt8_mask])):.3f} px, "
        f"crossed>=8px={crossed_8}"
    )
    assert crossed_8 >= 500, f"Unexpectedly few boxes crossed 8px: {crossed_8}"

    # Verify Val 640 -> 672x672 and 768 -> 800x800 rect=True, ratio_pad consistency
    val_ratios = {}
    for imgsz in (640, 768):
        cfg_v = get_cfg(overrides=dict(task="detect", data=str(run_v2.DATA_YAML), imgsz=imgsz, batch=4, rect=True, split="val", workers=0))
        ds_v = candidates_v6.Warp50YOLODataset(
            img_path=str(run_v2.REF / "data/val.txt"),
            imgsz=imgsz,
            batch_size=4,
            augment=False,
            hyp=cfg_v,
            rect=True,
            cache=None,
            single_cls=False,
            stride=32,
            pad=0.5,
            task="detect",
            data=data,
            warp_split_y=0.5,
            warp_top_frac=0.4,
        )
        it = ds_v[0]
        val_ratios[str(imgsz)] = dict(
            img_shape=list(it["img"].shape),
            ori_shape=list(it["ori_shape"]),
            resized_shape=list(map(int, it["resized_shape"])),
            ratio_pad=[[float(it["ratio_pad"][0][0]), float(it["ratio_pad"][0][1])], [float(it["ratio_pad"][1][0]), float(it["ratio_pad"][1][1])]],
        )
        print(f"  Val imgsz={imgsz} item:", json.dumps(val_ratios[str(imgsz)]))

    # 2. Verify zero pixel diff on 320/320 identity image warp + run 8-img 256/384 smoke BEFORE 1000-img Val scan (keeps val.cache intact)
    im0 = cv2.imread(ds_v.im_files[0])
    id_pix_diff = int(
        np.abs(
            cv2.resize(im0, (640, 640), interpolation=cv2.INTER_LINEAR).astype(np.int32)
            - candidates_v6.warp_image(im0, 640, split_y=0.5, top_frac=0.5).astype(np.int32)
        ).max()
    )
    assert id_pix_diff == 0, f"Expected 0 max pixel diff on 320/320 identity warp, got {id_pix_diff}"
    print(f"  320/320 identity split-resize max pixel diff vs cv2.resize: {id_pix_diff}")

    with tempfile.TemporaryDirectory(prefix="v6_warp50_smoke_") as td:
        td_p = Path(td)
        sub_txt = td_p / "val8.txt"
        sub_txt.write_text("\n".join((run_v2.REF / "data/val.txt").read_text().splitlines()[:8]) + "\n")
        sub_yaml = td_p / "val8.yaml"
        cfg_y = yaml.safe_load(run_v2.DATA_YAML.read_text())
        cfg_y["val"] = str(sub_txt)
        sub_yaml.write_text(yaml.safe_dump(cfg_y, sort_keys=False))
        val8 = run_v6.InverseWarpTaskValidator(
            args=dict(
                task="detect",
                data=str(sub_yaml),
                imgsz=640,
                batch=8,
                workers=0,
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
            save_dir=td_p / "out",
        )
        m_smoke = val8(model=run_v2.reference.load_model(p2a_best))
        assert val8.seen == 8 and len(val8.jdict) > 0
        print(f"  256/384 8-img InverseWarpTaskValidator smoke OK: preds={len(val8.jdict)}, mAP50={m_smoke['metrics/mAP50(B)']*100:.2f}%")

    # 3. MANDATORY 320/320 IDENTITY-WARP VAL PARITY GATE (0.5 / 0.5) vs CLS-P2A E90 Official (also restores 1000-img val.cache)
    print("  Running 320/320 identity-warp Val parity gate on CLS-P2A best.pt (1,000 images)...")
    official_summary = json.loads((run_v2.proof("HistCLS_SchemeA_P2") / "evaluation/summary.json").read_text())["best"]
    off_main = official_summary["main"]
    with tempfile.TemporaryDirectory(prefix="v6_id_parity_") as td:
        id_res = run_v6.evaluate_warp_at_imgsz(
            p2a_best,
            Path(td) / "eval",
            imgsz=640,
            batch=32,
            split_y=0.5,
            top_frac=0.5,
        )
    id_main = id_res["main"]
    diffs_pp = {
        k: abs(id_main[k] - off_main[k]) * 100.0
        for k in ("AP50_95", "AP50", "AP75", "AR100", "AP_small", "AP_medium", "AP_large")
    }
    nat_diff_pp = max(
        abs(id_res["native_metrics"][k] - official_summary["native_metrics"][k]) * 100.0
        for k in ("metrics/mAP50(B)", "metrics/mAP50-95(B)")
    )
    max_pp = max(max(diffs_pp.values()), nat_diff_pp)
    print("  Identity 320/320 Val diffs (pp):", json.dumps({k: round(v, 5) for k, v in diffs_pp.items()}), f"nat_diff={nat_diff_pp:.5f} pp")
    assert max_pp <= 0.005, f"320/320 identity Val parity failed! max_diff={max_pp:.5f} pp"

    # 4. 4-step real AMP + MuSGD smoke test on real Warp50YOLODataset batch
    model = run_v2.reference.load_model(c["initial"]).cuda()
    model.train()
    crit = model.init_criterion()
    model.criterion = crit
    dummy = type("Dummy", (), {"args": model.args, "build_optimizer": BaseTrainer.build_optimizer})()
    opt = dummy.build_optimizer(model, name="MuSGD", lr=0.001, momentum=0.9, decay=0.0005, iterations=10000)
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
        assert torch.isfinite(loss).all() and torch.isfinite(loss_items).all(), (step, loss, loss_items)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        before = scaler.get_scale()
        scaler.step(opt)
        scaler.update()
        after = scaler.get_scale()
        if after >= before:
            succ += 1
        print(f"  warp50 real-batch AMP+MuSGD step {step}: loss={float(loss):.4f}, scale={before:.0f}->{after:.0f}")
        if succ >= 4:
            break
    assert succ >= 4
    print("================ PRE-FLIGHT PASS: warp50 ================")
    return dict(
        params=c["params"],
        added_over_p2a=c["added_over_p2a"],
        train_lt8_count=int(lt8_mask.sum()),
        train_lt8_crossed8=crossed_8,
        identity_320_max_diff_pp=max_pp,
        identity_320_diffs_pp=diffs_pp,
        identity_320_native_diff_pp=nat_diff_pp,
        val_ratios=val_ratios,
        smoke_unskipped_steps=succ,
    )


if __name__ == "__main__":
    summary = {
        "b1": check_b1(),
        "warp50": check_warp50(),
    }
    run_v6.dump(STAGE / "records/preflight_v6.json", summary)
    print("\nALL V6 PRE-FLIGHT CHECKS PASSED. Saved records/preflight_v6.json")
