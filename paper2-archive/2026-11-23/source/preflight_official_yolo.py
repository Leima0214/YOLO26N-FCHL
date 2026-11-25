"""GPU pre-flight check for the 6 official YOLO horizontal arms: v8n, v9t, v10n, y11n, y12n, y26s."""
import json
import sys
import tempfile
from pathlib import Path

import cv2
import torch
import yaml
from ultralytics.cfg import get_cfg
from ultralytics.data.build import build_dataloader
from ultralytics.data.dataset import YOLODataset
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.utils.torch_utils import get_flops

STAGE = Path(__file__).resolve().parent
sys.path[:0] = [str(STAGE)]
import run_v2
import run_v7
import run_official_yolo as roy


def run_arm_preflight(arm: str, ds_train, sub_yaml: Path, td_p: Path, val_imgs) -> dict:
    assert not (STAGE / "runs" / arm).exists(), f"Existing run dir: runs/{arm}"
    c = roy.prepare(arm)
    opt_cfg = roy.OPTIM_SPECS[arm]

    # 1. 2-step real AMP optimizer smoke at batch=32
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    m = run_v2.reference.load_model(c["initial"]).cuda().train()
    for p in m.parameters():
        p.requires_grad_(True)
    m.args = get_cfg(
        overrides=dict(
            task="detect",
            imgsz=640,
            batch=32,
            epochs=100,
            optimizer=opt_cfg["optimizer"],
            lr0=opt_cfg["lr0"],
            lrf=opt_cfg["lrf"],
            momentum=opt_cfg["momentum"],
            weight_decay=opt_cfg["weight_decay"],
            amp=True,
            box=7.5,
            cls=0.5,
            dfl=1.5,
            nbs=64,
            seed=42,
        )
    )
    m.criterion = m.init_criterion()
    bt = object.__new__(BaseTrainer)
    bt.args = m.args
    bt.model = m
    opt = bt.build_optimizer(
        m,
        name=opt_cfg["optimizer"],
        lr=opt_cfg["lr0"],
        momentum=opt_cfg["momentum"],
        decay=opt_cfg["weight_decay"],
        iterations=18800,
    )
    assert type(opt).__name__ == opt_cfg["optimizer"], type(opt).__name__
    scaler = torch.amp.GradScaler("cuda", enabled=True)

    it = iter(build_dataloader(ds_train, batch=32, workers=2, shuffle=False, rank=-1))
    init_snap = {k: v.detach().clone() for k, v in m.state_dict().items()}
    losses = []
    for step_idx in range(2):
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

    changed = sum(1 for k, v in m.state_dict().items() if v.is_floating_point() and not torch.equal(v, init_snap[k]))
    peak_res = float(torch.cuda.max_memory_reserved() / 2**30)
    assert changed > 50 and peak_res < 20.0
    del opt, scaler, m, init_snap
    torch.cuda.empty_cache()

    # 2. 8-img COCO validator smoke (primary + secondary for dual-head arms)
    init_ck = Path(c["initial"])
    pmode = roy.PRIMARY_MODE[arm]
    coco_val_cls = roy.OfficialO2OCOCOValidator if pmode == "o2o" else roy.OfficialO2MCOCOValidator
    v_coco = coco_val_cls(
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
        save_dir=td_p / f"{arm}_coco",
    )
    m_coco = v_coco(model=run_v2.reference.load_model(init_ck))
    assert len(v_coco.evaluated_ids) == 8 and "COCO/AP50_95" in m_coco

    # Standalone evaluator smoke
    eval_fn = run_v7.evaluate_o2o_at_imgsz if pmode == "o2o" else roy.evaluate_o2m_official
    r_prim = eval_fn(init_ck, td_p / f"{arm}_prim", imgsz=640, batch=8, data_yaml=sub_yaml)
    sec_preds = None
    if arm in roy.DUAL_HEAD_ARMS:
        sec_fn = roy.evaluate_o2m_official if pmode == "o2o" else run_v7.evaluate_o2o_at_imgsz
        r_sec = sec_fn(init_ck, td_p / f"{arm}_sec", imgsz=640, batch=8, data_yaml=sub_yaml)
        sec_preds = r_sec["predictions"]

    # 3. Fuse & GFLOPs + single-image efficiency forward smoke
    raw_m = run_v2.reference.load_model(init_ck)
    dep = roy.compact_o2o(raw_m) if pmode == "o2o" else roy.safe_as_o2m(raw_m, fuse=True)
    dep_params = sum(p.numel() for p in dep.parameters())
    gflops = round(float(get_flops(dep.cpu(), imgsz=640)), 3)
    x_test = torch.zeros(1, 3, 640, 640, device="cuda")
    out_test = dep.cuda().eval()(x_test)
    assert out_test[0].shape[0] == 1
    del dep, raw_m
    torch.cuda.empty_cache()

    print(
        f"  [Preflight OK] {arm:<4s} ({roy.LABELS[arm]:<7s}): opt={opt_cfg['optimizer']} losses={losses[-1]} "
        f"changed={changed} vram={peak_res:.2f}GiB deploy={dep_params:,} gflops={gflops} prim_preds={r_prim['predictions']} sec_preds={sec_preds}"
    )
    return dict(
        arm=arm,
        label=roy.LABELS[arm],
        primary_mode=pmode,
        optimizer=opt_cfg["optimizer"],
        lr0=opt_cfg["lr0"],
        losses=losses,
        changed_tensors=changed,
        peak_reserved_GiB=round(peak_res, 3),
        train_params=c["params"],
        deployed_params=dep_params,
        gflops=gflops,
        smoke_primary_preds=r_prim["predictions"],
        smoke_secondary_preds=sec_preds,
    )


def main():
    print("================ PRE-FLIGHT OFFICIAL 6 YOLO BASELINES ================")
    data = yaml.safe_load(run_v2.DATA_YAML.read_text())
    hyp = get_cfg(overrides=dict(task="detect", data=str(run_v2.DATA_YAML), imgsz=640, batch=32, rect=False, mosaic=1.0, flipud=0.0, fliplr=0.5, workers=2, seed=42))
    ds_train = YOLODataset(
        img_path=str(run_v2.REF / "data/train.txt"),
        imgsz=640,
        batch_size=32,
        augment=True,
        hyp=hyp,
        rect=False,
        cache=None,
        single_cls=False,
        stride=32,
        pad=0.0,
        task="detect",
        data=data,
    )
    assert len(ds_train) == 6000
    val_lines = (run_v2.REF / "data/val.txt").read_text().splitlines()
    val_imgs = [cv2.imread(p) for p in val_lines[:4]]

    arms_out = {}
    with tempfile.TemporaryDirectory(prefix="official_yolo_preflight_") as td:
        td_p = Path(td)
        sub_txt = td_p / "val8.txt"
        sub_txt.write_text("\n".join(val_lines[:8]) + "\n")
        sub_yaml = td_p / "val8.yaml"
        cfg_y = yaml.safe_load(run_v2.DATA_YAML.read_text())
        cfg_y["val"] = str(sub_txt)
        sub_yaml.write_text(yaml.safe_dump(cfg_y, sort_keys=False))
        for arm in roy.ARMS:
            arms_out[arm] = run_arm_preflight(arm, ds_train, sub_yaml, td_p, val_imgs)

    # Restore full 1,000-image val.cache cleanly
    cfg_val = get_cfg(overrides=dict(task="detect", data=str(run_v2.DATA_YAML), imgsz=640, batch=32, rect=True, split="val", workers=0))
    ds_val = YOLODataset(
        img_path=str(run_v2.REF / "data/val.txt"),
        imgsz=640,
        batch_size=32,
        augment=False,
        hyp=cfg_val,
        rect=True,
        cache=None,
        single_cls=False,
        stride=32,
        pad=0.5,
        task="detect",
        data=data,
    )
    assert len(ds_val) == 1000 and sum(len(r["cls"]) for r in ds_val.labels) == 2612

    receipt = dict(
        time=run_v2.now(),
        passed=True,
        gpu=torch.cuda.get_device_name(0),
        arms=arms_out,
        full_val=dict(images=1000, instances=2612),
    )
    out_p = STAGE / "diagnostics/preflight_official_yolos.json"
    run_v2.dump(out_p, receipt)
    print(f"================ PRE-FLIGHT OFFICIAL 6 YOLOS PASSED -> {out_p} ================")


if __name__ == "__main__":
    main()
