"""V7 ablation candidates (no-EMA6, single-variable controls + sensitivity + 768 compute parity).

Arms (all seed=42, 100E, SVRDD7 Train 6000 / Val 1000, Test sealed):
  1) b0      : Native YOLO26n from verified 7-class B0_initial_seed42.pt (2,506,530 params, 640 uniform)
  2) b0_w50  : Native YOLO26n + #5 256/384 vertical resampling (exact same initial.pt as b0, 640)
  3) b0_p2   : P2-only detection head (HistoricalCLS minus paper2_pki2, 2,574,360 params, 640 uniform)
  4) b1_768  : B1 trained uniformly at imgsz=768 (exact same initial.pt as b1/b1_w50, 3,246,130 params)
  5) b1_w448 : B1 + #5 192/448 (split_y=0.5, top_frac=0.3, 1.4x lower road density, 640)
"""
import copy
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from ultralytics.nn.modules.head import Detect

from models import FusedDetect
import candidates_v6

STAGE = Path(__file__).resolve().parent
PUBLIC_YOLO26N = Path("/SVRDD/SVRDD_YOLO_READY/_paper2_archives/20261101_cls_inner_asff_freq_verified/external_initial/yolo26n.pt")
PUBLIC_YOLO26N_SHA = "9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef"
B0_INIT = Path("/root/YOLO26N-FCHL/reports/paper2_768_b0_p2_cls_set_20261106_42416/initialization/B0_initial_seed42.pt")
B0_INIT_SHA = "8c7d99e4143a0aa7"  # prefix checked + exact tensor parity vs yolo26n.pt

ARMS = (
    "b0",
    "b0_w50",
    "b0_p2",
    "b1_768",
    "b1_w448",
)
SEEDS = {arm: 42 for arm in ARMS}
IMGSZ = dict(
    b0=640,
    b0_w50=640,
    b0_p2=640,
    b1_768=768,
    b1_w448=640,
)
# None for uniform scaling; (split_y, top_frac) for single-breakpoint vertical resampling
WARP_CFGS = dict(
    b0=None,
    b0_w50=(0.5, 0.4),        # 256/384 at 640
    b0_p2=None,
    b1_768=None,
    b1_w448=(0.5, 0.3),       # 192/448 at 640
)
PROTOCOL_OVERRIDES = dict(
    b0={},
    b0_w50={},
    b0_p2={},
    b1_768={"imgsz": 768},
    b1_w448={},
)


def is_b0_family(arm: str) -> bool:
    return arm in ("b0", "b0_w50")


def is_b1_family(arm: str) -> bool:
    return arm in ("b1_768", "b1_w448")


def warp_cfg(arm: str) -> tuple[float, float] | None:
    return WARP_CFGS[arm]


class HistoricalP2OnlyDetect(FusedDetect):
    """P2 high-resolution head without PKI2 context (stride [4, 8, 16, 32], matching October HistoricalCLS)."""

    def forward(self, features):
        c2, p3, p4, p5 = features
        p2 = self.p2_fusion(torch.cat((c2, F.interpolate(p3, size=c2.shape[-2:], mode="nearest")), 1))
        return Detect.forward(self, [p2, p3, p4, p5, p2, p3, p4, p5])

    def forward_head(self, x, box_head=None, cls_head=None):
        if box_head is None or cls_head is None:
            return {}
        assert len(x) == 8 and self.nl == 4
        bs = x[0].shape[0]
        boxes = torch.cat([box_head[i](x[i]).view(bs, 4 * self.reg_max, -1) for i in range(4)], -1)
        scores = torch.cat([cls_head[i](x[i + 4]).view(bs, self.nc, -1) for i in range(4)], -1)
        return dict(boxes=boxes, scores=scores, feats=x[:4])


def compact_o2m(model, arm: str | None = None):
    """Build fused O2M deployment model for any Paper 2 arm (B0, B0+P2, CLS, CLS-P2A, or B1)."""
    from rebuilt_model import as_o2m

    out = as_o2m(model)
    head = out.model[-1]
    if hasattr(head, "ctx_o2o"):
        del head.ctx_o2o
    out.fuse(verbose=False)
    return out


def build(hist_base, arm: str, b0_ref=None):
    """Construct the exact initial model for arm with verified tensor inheritance."""
    if arm not in ARMS:
        raise ValueError(f"Unknown v7 arm: {arm}")

    if is_b0_family(arm):
        assert b0_ref is not None
        model = copy.deepcopy(b0_ref)
        head = model.model[-1]
        assert type(head).__name__ == "Detect" and head.nl == 3 and head.end2end
        assert head.stride.tolist() == [8.0, 16.0, 32.0]
        return model

    if arm == "b0_p2":
        model = copy.deepcopy(hist_base)
        head = model.model[-1]
        assert hasattr(head, "p2_fusion") and hasattr(head, "paper2_pki2")
        del head.paper2_pki2
        head.__class__ = HistoricalP2OnlyDetect
        return model

    if is_b1_family(arm):
        return candidates_v6.build(hist_base, "b1")

    raise AssertionError(arm)


@torch.no_grad()
def prepare(hist_base, b0_ref, pub_yolo26n, arm: str, file_sha_fn, state_sha_fn):
    """Run all initialization, tensor-equality, fusion parity, and warp geometry checks for arm."""
    assert file_sha_fn(PUBLIC_YOLO26N) == PUBLIC_YOLO26N_SHA, "Public yolo26n.pt SHA mismatch!"
    assert file_sha_fn(B0_INIT).startswith(B0_INIT_SHA), f"B0_INIT SHA mismatch: {file_sha_fn(B0_INIT)}"

    candidate = build(hist_base, arm, b0_ref=b0_ref)
    cand_sd = candidate.state_dict()
    checks: dict[str, Any] = {}

    if is_b0_family(arm):
        pub_sd = pub_yolo26n.state_dict()
        assert len(cand_sd) == len(pub_sd) == 708 and set(cand_sd) == set(pub_sd)
        shape_eq = [k for k in cand_sd if cand_sd[k].shape == pub_sd[k].shape]
        shape_diff = [k for k in cand_sd if cand_sd[k].shape != pub_sd[k].shape]
        assert len(shape_eq) == 606 and len(shape_diff) == 102
        assert all(torch.equal(cand_sd[k], pub_sd[k]) for k in shape_eq)
        assert all(k.startswith(("model.23.cv3.", "model.23.one2one_cv3.")) for k in shape_diff)
        b0_sd = b0_ref.state_dict()
        assert all(torch.equal(cand_sd[k], b0_sd[k]) for k in b0_sd)
        n = sum(p.numel() for p in candidate.parameters())
        assert n == 2506530, n
        from rebuilt_model import as_o2m

        u_o2m = as_o2m(copy.deepcopy(candidate))
        unf_n = sum(p.numel() for p in u_o2m.parameters())
        assert unf_n == 2384577, unf_n
        f_o2m = compact_o2m(candidate, arm)
        dep_n = sum(p.numel() for p in f_o2m.parameters())
        assert dep_n == 2376201, dep_n
        checks["matches_public_yolo26n_all_606_tensors"] = True
        checks["matches_b0_init_exact"] = True
        checks["unfused_o2m_params"] = unf_n
        checks["deployed_o2m_params"] = dep_n
        checks["b0_end2end_eval_shape_640"] = list(copy.deepcopy(candidate).eval()(torch.zeros(1, 3, 640, 640))[0].shape)
        checks["b0_o2m_eval_shape_640"] = list(f_o2m(torch.zeros(1, 3, 640, 640))[0].shape)

    elif arm == "b0_p2":
        hist_sd = hist_base.state_dict()
        # Every single parameter/buffer outside paper2_pki2 must equal HistoricalCLS bit-for-bit
        for k, v in cand_sd.items():
            assert "paper2_pki2" not in k
            assert k in hist_sd and torch.equal(v, hist_sd[k]), f"b0_p2 tensor mismatch vs HistoricalCLS on {k}"
        n = sum(p.numel() for p in candidate.parameters())
        assert n == 2574360, n
        bns = [(m.eps, m.momentum) for m in candidate.model[-1].p2_fusion.modules() if isinstance(m, nn.BatchNorm2d)]
        assert bns == [(0.001, 0.03)] * 4, bns
        from rebuilt_model import as_o2m

        x = torch.rand(1, 3, 128, 160, generator=torch.Generator().manual_seed(43))
        u_o2m = as_o2m(copy.deepcopy(candidate))
        f_o2m = compact_o2m(copy.deepcopy(candidate), arm)
        yu, yf = u_o2m(x)[0], f_o2m(x)[0]
        box_err = float((yu[:, :4] - yf[:, :4]).abs().max())
        score_err = float((yu[:, 4:] - yf[:, 4:]).abs().max())
        assert box_err < 5e-3 and score_err < 1e-4, (box_err, score_err)
        checks["inherits_hist_non_pki_exact"] = True
        checks["p2_fusion_bn_count"] = len(bns)
        checks["active_fuse_box_err_px"] = box_err
        checks["active_fuse_score_err"] = score_err
        checks["deployed_o2m_params"] = sum(p.numel() for p in f_o2m.parameters())

    elif is_b1_family(arm):
        # Strictly verify byte-identical initial state vs frozen V6 b1
        cand_v6, c_v6 = candidates_v6.prepare(hist_base, "b1")
        assert state_sha_fn(candidate) == state_sha_fn(cand_v6)
        n = sum(p.numel() for p in candidate.parameters())
        assert n == 3246130, n
        checks.update(c_v6["checks"])
        checks["matches_b1_prepare_exact"] = True

    wcfg = warp_cfg(arm)
    if wcfg is not None:
        split_y, top_frac = wcfg
        top_h, bot_h, eff_top = candidates_v6.effective_warp_frac(IMGSZ[arm], split_y, top_frac)
        assert top_h % 32 == 0 and bot_h % 32 == 0, (top_h, bot_h)
        ys = np.array([0.0, 0.25, 0.499, 0.5, 0.501, 0.75, 1.0, -0.05, 1.05], dtype=np.float64)
        rt_np = float(np.max(np.abs(candidates_v6.warp_y_inverse_np(candidates_v6.warp_y_forward_np(ys, split_y, eff_top), split_y, eff_top) - ys)))
        rt_t = float(
            (
                candidates_v6.warp_y_inverse_torch(
                    torch.tensor(candidates_v6.warp_y_forward_np(ys, split_y, eff_top)), split_y, eff_top
                )
                - torch.tensor(ys)
            )
            .abs()
            .max()
        )
        assert max(rt_np, rt_t) < 1e-12, (rt_np, rt_t)
        checks["warp_cfg"] = dict(split_y=split_y, top_frac=top_frac, top_h=top_h, bot_h=bot_h, eff_top_frac=eff_top)
        checks["warp_roundtrip_max_err"] = float(max(rt_np, rt_t))

    parent_desc = (
        f"PublicYOLO26n+B0_initial({B0_INIT.name})"
        if is_b0_family(arm)
        else ("HistoricalCLS minus PKI2" if arm == "b0_p2" else "CLS-P2A+B1 (records/b1/initial.pt)")
    )
    return candidate, dict(
        arm=arm,
        seed=SEEDS[arm],
        imgsz=IMGSZ[arm],
        parent_desc=parent_desc,
        params=sum(p.numel() for p in candidate.parameters()),
        warp=wcfg,
        protocol_overrides=PROTOCOL_OVERRIDES[arm],
        checks=checks,
    )
