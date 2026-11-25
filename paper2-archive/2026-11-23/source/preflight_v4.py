"""Strict GPU pre-flight checks for all 3 V4 arms before launching 100E training."""
import copy
import subprocess
import sys
from pathlib import Path
import torch

STAGE = Path("/root/YOLO26N-FCHL/reports/paper2_cls_candidate_diagnosis_640_20261120_42106")
sys.path[:0] = [str(STAGE)]
import run_v4
import candidates_v4
from ultralytics.engine.trainer import BaseTrainer


def check_arm(arm: str):
    print(f"\n================ PRE-FLIGHT V4: {arm} ================")
    c = run_v4.prepare(arm)
    model = run_v4.reference.load_model(c["initial"]).cuda()
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
        if any(k in name for k in ("context.", "context_p3.", "ctx_o2m", "ctx_o2o", "ctx_p3_o2m", "ctx_p3_o2o")):
            new_names.append(name)
            gi, um, wd = id_to_group[id(p)]
            print(f"  param {name:<44} shape={str(tuple(p.shape)):<18} group={gi} muon={um} wd={wd}")
            if p.ndim == 1:
                assert not um, f"1D param {name} wrongly assigned muon={um}"

    expected_new = dict(HistoricalCLS_seed100=0, HistCLS_SchemeA_P2_seed100=6, HistCLS_SchemeA_P2P3=12)[arm]
    assert len(new_names) == expected_new, f"Expected {expected_new} new tensors, got {len(new_names)}: {new_names}"

    # 2. Run real AMP + MuSGD + clip_grad_norm_(10.0) steps until 4 unskipped updates
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
    for name in new_names:
        p = dict(model.named_parameters())[name]
        delta = float((p.detach() - init_snap[name]).abs().max())
        print(f"  updated {name:<42} max_abs_delta={delta:.6e}")
        assert delta > 0.0, f"Parameter {name} did NOT update after 4 AMP+MuSGD steps!"

    # 3. Verify FP32 post-update as_o2m vs compact_o2m parity on BOTH 480x640 AND 768x768
    prev_tf32_mm = torch.backends.cuda.matmul.allow_tf32
    prev_tf32_cn = torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        from rebuilt_model import as_o2m

        model.eval()
        u_o2m = as_o2m(copy.deepcopy(model)).cuda().eval()
        f_o2m = candidates_v4.compact_o2m(copy.deepcopy(model)).cuda().eval()
        for sh in ((2, 3, 480, 640), (1, 3, 768, 768)):
            with torch.no_grad():
                x_val = torch.rand(*sh, device="cuda", generator=torch.Generator(device="cuda").manual_seed(9))
                yu = u_o2m(x_val)[0]
                yf = f_o2m(x_val)[0]
            box_err = float((yu[:, :4] - yf[:, :4]).abs().max())
            score_err = float((yu[:, 4:] - yf[:, 4:]).abs().max())
            print(f"  post-update FP32 fuse parity {sh[2]}x{sh[3]}: box_err={box_err:.3e} px, score_err={score_err:.3e}")
            assert box_err < 1e-2 and score_err < 1e-4, (sh, box_err, score_err)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev_tf32_mm
        torch.backends.cudnn.allow_tf32 = prev_tf32_cn

    # 4. Verify fresh-process unpickle (diagnose.py compatibility)
    probe = (
        f"import sys; sys.path[:0]=['{STAGE}']; from run import reference; "
        f"m = reference.load_model('{c['initial']}'); "
        f"print('  fresh-process unpickle OK:', type(m.model[-1]).__name__, sum(p.numel() for p in m.parameters()))"
    )
    res = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=True)
    print(res.stdout.strip())
    print(f"  protocol verified: seed={c['seed']}, close_mosaic={c['protocol']['close_mosaic']}, epochs={c['protocol']['epochs']}")
    print(f"================ PRE-FLIGHT PASS: {arm} ================")


if __name__ == "__main__":
    for arm in candidates_v4.ARMS:
        check_arm(arm)
    print("\nALL V4 PRE-FLIGHT CHECKS PASSED.")
