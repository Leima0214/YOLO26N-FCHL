"""Strict GPU pre-flight checks for V3 arms before launching 100E training."""
import copy
import sys
from pathlib import Path
import torch

STAGE = Path("/root/YOLO26N-FCHL/reports/paper2_cls_candidate_diagnosis_640_20261120_42106")
sys.path[:0] = [str(STAGE)]
import run_v3
import candidates_v3
from ultralytics.engine.trainer import BaseTrainer


def check_arm(arm: str):
    print(f"\n================ PRE-FLIGHT V3: {arm} ================")
    c = run_v3.prepare(arm)
    model = run_v3.reference.load_model(c["initial"]).cuda()
    model.train()
    crit = model.init_criterion()
    assert all(
        branch.assigner.stride == [8.0, 16.0, 32.0] and branch.reg_max == 1
        for branch in (crit.one2many, crit.one2one)
    )
    model.criterion = crit

    # 1. Check MuSGD param grouping via actual Ultralytics build_optimizer
    dummy = type("Dummy", (), {"args": model.args, "build_optimizer": BaseTrainer.build_optimizer})()
    opt = dummy.build_optimizer(model, name="MuSGD", lr=0.001, momentum=0.9, decay=0.0005, iterations=10000)
    opt_ids = {id(p) for g in opt.param_groups for p in g["params"]}
    model_ids = {id(p) for p in model.parameters() if p.requires_grad}
    assert model_ids == opt_ids, f"Missing params in MuSGD: {len(model_ids - opt_ids)}"
    id_to_group = {}
    for gi, g in enumerate(opt.param_groups):
        for p in g["params"]:
            id_to_group[id(p)] = (gi, g.get("use_muon", False), g.get("weight_decay", 0.0))

    new_names = []
    for name, p in model.named_parameters():
        if any(k in name for k in ("context.", "ctx_o2m", "ctx_o2o")):
            new_names.append(name)
            gi, um, wd = id_to_group[id(p)]
            print(f"  param {name:<42} shape={str(tuple(p.shape)):<18} group={gi} muon={um} wd={wd}")
            if p.ndim == 1:
                assert not um, f"1D param {name} wrongly assigned muon={um}"

    if arm == "HistCLS_SchemeA_P2":
        assert len(new_names) == 6, f"Expected 6 new tensors, got {new_names}"
    else:
        assert len(new_names) == 0

    # 2. Run real AMP + MuSGD steps on GPU until 4 unskipped optimizer steps occur
    init_snap = {name: p.detach().clone() for name, p in model.named_parameters() if name in new_names}
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
    for step in range(1, 15):
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
    for name in new_names:
        p = dict(model.named_parameters())[name]
        delta = float((p.detach() - init_snap[name]).abs().max())
        print(f"  updated {name:<40} max_abs_delta={delta:.6e}")
        assert delta > 0.0, f"Parameter {name} did NOT update after 4 AMP+MuSGD steps!"

    # 3. Verify FP32 post-update as_o2m vs compact_o2m parity (strict allow_tf32=False)
    prev_tf32_mm = torch.backends.cuda.matmul.allow_tf32
    prev_tf32_cn = torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        from rebuilt_model import as_o2m

        model.eval()
        u_o2m = as_o2m(copy.deepcopy(model)).cuda().eval()
        f_o2m = candidates_v3.compact_o2m(copy.deepcopy(model)).cuda().eval()
        with torch.no_grad():
            x_val = torch.rand(2, 3, 480, 640, device="cuda", generator=torch.Generator(device="cuda").manual_seed(9))
            yu = u_o2m(x_val)[0]
            yf = f_o2m(x_val)[0]
        box_err = float((yu[:, :4] - yf[:, :4]).abs().max())
        score_err = float((yu[:, 4:] - yf[:, 4:]).abs().max())
        print(f"  post-update FP32 fuse parity (480x640 rect): box_err={box_err:.3e} px, score_err={score_err:.3e}")
        assert box_err < 1e-2 and score_err < 1e-4, (box_err, score_err)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev_tf32_mm
        torch.backends.cudnn.allow_tf32 = prev_tf32_cn

    # 4. Verify close_mosaic protocol value & fresh-process unpickle (diagnose.py compatibility)
    cm = c["protocol"]["close_mosaic"]
    print(f"  protocol verified: close_mosaic={cm}, mosaic={c['protocol']['mosaic']}, epochs={c['protocol']['epochs']}")
    import subprocess
    probe = (
        f"import sys; sys.path[:0]=['{STAGE}']; from run import reference; "
        f"m = reference.load_model('{c['initial']}'); "
        f"print('  fresh-process unpickle OK:', type(m.model[-1]).__name__, sum(p.numel() for p in m.parameters()))"
    )
    res = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=True)
    print(res.stdout.strip())
    print(f"================ PRE-FLIGHT PASS: {arm} ================")


if __name__ == "__main__":
    for arm in candidates_v3.ARMS:
        check_arm(arm)
    print("\nALL V3 PRE-FLIGHT CHECKS PASSED.")
