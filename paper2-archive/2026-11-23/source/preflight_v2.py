"""Strict pre-flight checks for V2 arms before launching 100E GPU training."""
import copy
import json
import sys
from pathlib import Path
import torch

STAGE = Path("/root/YOLO26N-FCHL/reports/paper2_cls_candidate_diagnosis_640_20261120_42106")
sys.path[:0] = [str(STAGE)]
import run_v2
import candidates_v2
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.utils.torch_utils import unwrap_model


def check_arm(arm: str):
    print(f"\n================ PRE-FLIGHT: {arm} ================")
    c = run_v2.prepare(arm)
    model = run_v2.reference.load_model(c["initial"]).cuda()
    model.train()
    crit = model.init_criterion()
    for branch in (crit.one2many, crit.one2one):
        branch.assigner.stride = [8.0, 16.0, 32.0]
    model.criterion = crit

    # 1. Check MuSGD param grouping via actual Ultralytics build_optimizer
    dummy = type("Dummy", (), {"args": model.args, "build_optimizer": BaseTrainer.build_optimizer})()
    opt = dummy.build_optimizer(model, name="MuSGD", lr=0.01, momentum=0.9, decay=0.0005, iterations=10000)
    opt_ids = {id(p) for g in opt.param_groups for p in g["params"]}
    model_ids = {id(p) for p in model.parameters() if p.requires_grad}
    assert model_ids == opt_ids, f"Missing params in MuSGD: {len(model_ids - opt_ids)}"
    # Verify 1D params (alpha / gamma / bias) are NOT in Muon (use_muon=True) and NOT weight-decayed
    id_to_group = {}
    for gi, g in enumerate(opt.param_groups):
        for p in g["params"]:
            id_to_group[id(p)] = (gi, g.get("use_muon", False), g.get("weight_decay", 0.0))
    for name, p in model.named_parameters():
        if any(k in name for k in ("reg_qual_o2m", "alpha", "gamma", "gate")):
            gi, um, wd = id_to_group[id(p)]
            print(f"  param {name:<42} shape={str(tuple(p.shape)):<18} group={gi} muon={um} wd={wd}")
            if p.ndim == 1:
                assert not um, f"1D param {name} wrongly assigned muon={um}"

    # 2. Check 3 real AMP forward+backward+MuSGD steps on valid [0,1] image batch
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    x = torch.rand(4, 3, 640, 640, device="cuda")
    batch = {
        "batch_idx": torch.tensor([0, 0, 1, 2, 3], dtype=torch.float32, device="cuda"),
        "cls": torch.tensor([[0], [1], [4], [2], [3]], dtype=torch.float32, device="cuda"),
        "bboxes": torch.tensor(
            [
                [0.5, 0.5, 0.02, 0.30],
                [0.3, 0.4, 0.25, 0.015],
                [0.6, 0.6, 0.04, 0.04],
                [0.5, 0.5, 0.30, 0.30],
                [0.2, 0.7, 0.06, 0.05],
            ],
            dtype=torch.float32,
            device="cuda",
        ),
    }
    snap0 = {n: p.detach().clone() for n, p in model.named_parameters() if any(k in n for k in ("reg_qual_o2m", "alpha", "gamma", "gate"))}
    success_steps = 0
    for step in range(10):
        opt.zero_grad()
        with torch.amp.autocast("cuda", enabled=True):
            preds = model(x)
            loss, _ = crit(preds, batch)
            total = loss.sum()
        assert torch.isfinite(total), f"Non-finite loss at step {step}: {loss}"
        scaler.scale(total).backward()
        scaler.unscale_(opt)
        bad = [n for n, p in model.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()]
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        before = scaler.get_scale()
        scaler.step(opt)
        scaler.update()
        after = scaler.get_scale()
        if after >= before:
            success_steps += 1
        print(f"  step {step}: loss={float(total):.4f}, scale {before:.0f}->{after:.0f}, bad_grads={bad[:5]}")
        if success_steps >= 3:
            break
    assert success_steps >= 3, f"Did not reach 3 successful AMP steps on {arm}!"
    moved = {}
    for n, p0 in snap0.items():
        diff = float((model.state_dict()[n] - p0).abs().max())
        moved[n] = diff
    print("  3-step AMP+MuSGD max param updates:", {k: round(v, 7) for k, v in moved.items()})
    assert all(v > 0 and torch.isfinite(model.state_dict()[k]).all() for k, v in moved.items()), f"Update check failed: {moved}"

    # 3. Check model.eval() + as_o2m() + fuse() in strict FP32 (matching TaskValidator!)
    from rebuilt_model import as_o2m
    prev_tf32_mm, prev_tf32_cd = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    m_eval = as_o2m(model.cpu()).cuda().eval()
    m_fused = candidates_v2.compact_o2m(model.cpu()).cuda().eval()
    with torch.no_grad():
        x_val = torch.rand(2, 3, 640, 640, device="cuda")
        y1 = m_eval(x_val)[0]
        y2 = m_fused(x_val)[0]
        box_err = float((y1[:, :4] - y2[:, :4]).abs().max())
        score_err = float((y1[:, 4:] - y2[:, 4:]).abs().max())
        assert y1.shape == (2, 11, 34000) and box_err < 1e-2 and score_err < 1e-4, (
            f"Eval/fuse mismatch: shape={y1.shape}, box_err={box_err}, score_err={score_err}"
        )
    torch.backends.cuda.matmul.allow_tf32 = prev_tf32_mm
    torch.backends.cudnn.allow_tf32 = prev_tf32_cd
    print(f"  [PASS] {arm}: shape={tuple(y1.shape)}, box_err={box_err:.6f}px, score_err={score_err:.6f}")


if __name__ == "__main__":
    for arm in ("HistCLS_O2M_RegQual", "HistCLS_AntiAlias_L01"):
        check_arm(arm)
    print("\nALL HISTORICAL-CLS V2 PRE-FLIGHT CHECKS PASSED!")
