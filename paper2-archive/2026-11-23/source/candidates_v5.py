"""V5 candidate architectures built strictly from the original October HistoricalCLS state.
Arms:
  1) p2_det : HistoricalCLS + Scheme A P2 context with .detach() on [p2, p3] inputs, seed=42 (+12,376 params)
  2) p2_cm0 : HistoricalCLS + Scheme A P2 context with close_mosaic=0 across 100E, seed=42 (+12,376 params)
"""
import copy
import torch
from torch import nn
import torch.nn.functional as F
from ultralytics.nn.modules.head import Detect
from historical_head import P2PKI2ClsDetect
from candidates_v3 import HistoricalScoreContextDetect

ARMS = (
    "p2_det",
    "p2_cm0",
)
SEEDS = dict(
    p2_det=42,
    p2_cm0=42,
)
ADDED = dict(
    p2_det=12376,
    p2_cm0=12376,
)
PROTOCOL_OVERRIDES = dict(
    p2_det={},
    p2_cm0={"close_mosaic": 0},
)


class HistoricalScoreContextDetachDetect(HistoricalScoreContextDetect):
    """Identical to HistoricalScoreContextDetect except upstream inputs [p2, p3] are detached before context."""

    def forward(self, features):
        c2, p3, p4, p5 = features
        context_pki = self.paper2_pki2([p3, p4, p5])
        p2 = self.p2_fusion(torch.cat((c2, F.interpolate(p3, size=c2.shape[-2:], mode="nearest")), 1))
        p2_ctx = self.context(torch.cat((F.pixel_unshuffle(p2.detach(), 2), p3.detach()), 1))
        boxes = [p2, p3, p4, p5]
        classes = [p2, *context_pki]
        return Detect.forward(self, boxes + classes + [p2_ctx])


def build(parent, arm: str):
    if arm not in ARMS:
        raise ValueError(f"Unknown v5 arm: {arm}")
    assert isinstance(parent.model[-1], P2PKI2ClsDetect), (
        f"Expected HistoricalCLS P2PKI2ClsDetect, got {type(parent.model[-1])}"
    )
    model = copy.deepcopy(parent)
    head = model.model[-1]
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(SEEDS[arm])
        head.__class__ = HistoricalScoreContextDetachDetect if arm == "p2_det" else HistoricalScoreContextDetect
        head.context = nn.Sequential(
            nn.Conv2d(320, 32, 1, bias=False),
            nn.SiLU(),
            nn.Conv2d(32, 32, 3, padding=1, groups=32, bias=False),
            nn.SiLU(),
        )
        head.ctx_o2m = nn.Conv2d(32, 28, 1)
        head.ctx_o2o = nn.Conv2d(32, 28, 1)
        for m in (head.ctx_o2m, head.ctx_o2o):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)
    return model


def compact_o2m(model):
    from rebuilt_model import as_o2m

    out = as_o2m(model)
    head = out.model[-1]
    if isinstance(head, HistoricalScoreContextDetect) and hasattr(head, "ctx_o2o"):
        del head.ctx_o2o
    out.fuse(verbose=False)
    return out


@torch.no_grad()
def prepare(parent, arm: str):
    candidate = build(parent, arm)
    old = parent.state_dict()
    new = candidate.state_dict()
    assert all(k in new and torch.equal(v, new[k]) for k, v in old.items()), (
        "HistoricalCLS parent weight inheritance broken!"
    )
    n = sum(p.numel() for p in candidate.parameters())
    assert n == 3195674 + ADDED[arm], f"Param mismatch: got {n}, expected {3195674 + ADDED[arm]}"

    # 1. Step-0 exact equivalence on both 128x160 and 416x640 vs HistoricalCLS
    a, b = copy.deepcopy(parent).eval(), copy.deepcopy(candidate).eval()
    checks = {}
    for shape_tag, (hh, ww) in (("rect_128x160", (128, 160)), ("rect_416x640", (416, 640))):
        x = torch.rand(1, 3, hh, ww, generator=torch.Generator().manual_seed(43))
        ra, rb = a(x)[1], b(x)[1]
        c_sh = {
            branch: {k: float((ra[branch][k] - rb[branch][k]).abs().max()) for k in ("boxes", "scores")}
            for branch in ("one2many", "one2one")
        }
        assert all(val == 0.0 for m in c_sh.values() for val in m.values()), (shape_tag, c_sh)
        checks[f"step0_{shape_tag}"] = c_sh

    # 2. Active non-zero P2-only isolation & deployment fuse parity check
    x = torch.rand(1, 3, 128, 160, generator=torch.Generator().manual_seed(43))
    c_active = copy.deepcopy(candidate).eval()
    gen = torch.Generator().manual_seed(99)
    c_active.model[-1].ctx_o2m.weight.normal_(0, 0.1, generator=gen)
    c_active.model[-1].ctx_o2m.bias.normal_(0, 0.1, generator=gen)
    c_active.model[-1].ctx_o2o.weight.normal_(0, 0.1, generator=gen)
    c_active.model[-1].ctx_o2o.bias.normal_(0, 0.1, generator=gen)
    r_par = a(x)[1]["one2many"]
    r_act = c_active(x)[1]["one2many"]
    n_p2 = (128 // 4) * (160 // 4)
    p2_diff = float((r_act["scores"][:, :, :n_p2] - r_par["scores"][:, :, :n_p2]).abs().max())
    p345_diff = float((r_act["scores"][:, :, n_p2:] - r_par["scores"][:, :, n_p2:]).abs().max())
    box_diff = float((r_act["boxes"] - r_par["boxes"]).abs().max())
    assert p2_diff > 1e-3, f"Expected active P2 delta, got {p2_diff}"
    assert p345_diff == 0.0, f"P3/P4/P5 scores altered: {p345_diff}"
    assert box_diff == 0.0, f"Boxes altered: {box_diff}"
    checks["active_isolation"] = dict(p2_diff=p2_diff, p345_diff=p345_diff, box_diff=box_diff)

    from rebuilt_model import as_o2m

    u_o2m = as_o2m(copy.deepcopy(c_active))
    f_o2m = compact_o2m(copy.deepcopy(c_active))
    yu, yf = u_o2m(x)[0], f_o2m(x)[0]
    fuse_box_err = float((yu[:, :4] - yf[:, :4]).abs().max())
    fuse_score_err = float((yu[:, 4:] - yf[:, 4:]).abs().max())
    assert fuse_box_err < 5e-3 and fuse_score_err < 1e-4, (fuse_box_err, fuse_score_err)
    checks["active_fuse_box_err_px"] = fuse_box_err
    checks["active_fuse_score_err"] = fuse_score_err
    checks["deployed_o2m_params"] = sum(p.numel() for p in f_o2m.parameters())

    # 3. Gradient flow, regression isolation, O2O detach isolation, and p2_det upstream detach isolation
    with torch.enable_grad():
        xg = torch.rand(2, 3, 128, 128, generator=torch.Generator().manual_seed(7))
        # 3a. Step-1 gradient into zero-init ctx_o2m and ctx_o2o
        g1 = copy.deepcopy(candidate).train()
        out1 = g1(xg)
        loss1 = (out1["one2many"]["scores"] ** 2).mean() + (out1["one2one"]["scores"] ** 2).mean()
        loss1.backward()
        o2m_w = float(g1.model[-1].ctx_o2m.weight.grad.abs().max())
        o2m_b = float(g1.model[-1].ctx_o2m.bias.grad.abs().max())
        o2o_w = float(g1.model[-1].ctx_o2o.weight.grad.abs().max())
        o2o_b = float(g1.model[-1].ctx_o2o.bias.grad.abs().max())
        assert min(o2m_w, o2m_b, o2o_w, o2o_b) > 0.0, (o2m_w, o2m_b, o2o_w, o2o_b)
        checks["step1_proj_grad"] = dict(o2m_w=o2m_w, o2m_b=o2m_b, o2o_w=o2o_w, o2o_b=o2o_b)

        # 3b. Step-2 gradient into context[0] and context[2] + cv2 isolation
        g2 = copy.deepcopy(candidate).train()
        g2.model[-1].ctx_o2m.weight.data.fill_(0.05)
        out2 = g2(xg)["one2many"]
        out2["scores"].sum().backward()
        c0_g = float(g2.model[-1].context[0].weight.grad.abs().max())
        c2_g = float(g2.model[-1].context[2].weight.grad.abs().max())
        assert c0_g > 0.0 and c2_g > 0.0, (c0_g, c2_g)
        cv2_max = max(
            (float(p.grad.abs().max()) if p.grad is not None else 0.0)
            for p in g2.model[-1].cv2.parameters()
        )
        assert cv2_max == 0.0, f"Gradient leaked into cv2: {cv2_max}"
        checks["step2_context_grad"] = dict(pw=c0_g, dw=c2_g)
        checks["reg_isolation_max_cv2_grad"] = 0.0

        # 3c. Strict O2O detach isolation
        g3 = copy.deepcopy(candidate).train()
        g3.model[-1].ctx_o2o.weight.data.fill_(0.05)
        out3 = g3(xg)["one2one"]
        out3["scores"].sum().backward()
        o2o_leak = max(
            (float(p.grad.abs().max()) if p.grad is not None else 0.0)
            for p in list(g3.model[-1].context.parameters()) + list(g3.model[-1].p2_fusion.parameters())
        )
        assert o2o_leak == 0.0, f"O2O leaked gradient into context/p2_fusion: {o2o_leak}"
        checks["o2o_detach_max_shared_grad"] = 0.0

        # 3d. Exclusive p2_det upstream detach isolation check
        if arm == "p2_det":
            g_det = copy.deepcopy(candidate).train()
            head_det = g_det.model[-1]
            head_det.ctx_o2m.weight.data.fill_(0.05)
            captured = {}

            def _hook(_mod, _inp, out):
                captured["p2_ctx"] = out

            handle = head_det.context.register_forward_hook(_hook)
            try:
                _ = g_det(xg)
            finally:
                handle.remove()

            p2_ctx_out = captured["p2_ctx"]
            res_loss = F.pixel_shuffle(head_det.ctx_o2m(p2_ctx_out), 2).sum()
            res_loss.backward()

            dc0 = float(head_det.context[0].weight.grad.abs().max())
            dc2 = float(head_det.context[2].weight.grad.abs().max())
            dproj = float(head_det.ctx_o2m.weight.grad.abs().max())
            assert dc0 > 0.0 and dc2 > 0.0 and dproj > 0.0, (dc0, dc2, dproj)

            for mname, mod in (
                ("p2_fusion", head_det.p2_fusion),
                ("paper2_pki2", head_det.paper2_pki2),
                ("cv2", head_det.cv2),
                ("cv3", head_det.cv3),
                ("backbone_neck", g_det.model[:-1]),
            ):
                leak = max((float(p.grad.abs().max()) if p.grad is not None else 0.0) for p in mod.parameters())
                assert leak == 0.0, f"Detach isolation broken — leaked into {mname}: {leak}"
            checks["detach_in_residual_self_grad"] = dict(pw=dc0, dw=dc2, o2m_w=dproj)
            checks["detach_in_max_upstream_grad"] = 0.0

    return candidate, dict(
        arm=arm,
        seed=SEEDS[arm],
        params=n,
        added=ADDED[arm],
        inherited_tensors=len(old),
        protocol_overrides=PROTOCOL_OVERRIDES[arm],
        checks=checks,
    )
