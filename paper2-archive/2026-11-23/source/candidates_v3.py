"""V3 candidate architectures built strictly from the original October HistoricalCLS state."""
import copy
import torch
from torch import nn
import torch.nn.functional as F
from ultralytics.nn.modules.head import Detect
from historical_head import P2PKI2ClsDetect

ARMS = (
    "HistCLS_SchemeA_P2",
    "HistCLS_CloseMosaic0",
)
ADDED = dict(
    HistCLS_SchemeA_P2=12376,
    HistCLS_CloseMosaic0=0,
)
PROTOCOL_OVERRIDES = dict(
    HistCLS_SchemeA_P2={},
    HistCLS_CloseMosaic0={"close_mosaic": 0},
)


class HistoricalScoreContextDetect(P2PKI2ClsDetect):
    """Original October P2PKI2ClsDetect + Scheme A (P2-only classification score context).

    Computes lightweight multi-scale context from [pixel_unshuffle(p2, 2), p3] (320 -> 32 @ stride 8)
    and projects a zero-initialized pixel-shuffled logit residual strictly onto P2 classification scores.
    - Preserves P2PKI2ClsDetect p2_fusion (BN eps=0.001, momentum=0.03) and HistoricalPKI verbatim.
    - Zero change to regression towers (cv2 / one2one_cv2) or P3/P4/P5 classification logits.
    - Detect.forward detaches all 9 streams before one2one, preserving strict O2O gradient isolation.
    - Exact 0.0 step-0 equivalence with original October HistoricalCLS on both O2M and O2O.
    """

    def forward(self, features):
        c2, p3, p4, p5 = features
        context_pki = self.paper2_pki2([p3, p4, p5])
        p2 = self.p2_fusion(torch.cat((c2, F.interpolate(p3, size=c2.shape[-2:], mode="nearest")), 1))
        p2_ctx = self.context(torch.cat((F.pixel_unshuffle(p2, 2), p3), 1))
        boxes = [p2, p3, p4, p5]
        classes = [p2, *context_pki]
        return Detect.forward(self, boxes + classes + [p2_ctx])

    def forward_head(self, x, box_head=None, cls_head=None):
        result = super().forward_head(x[:8], box_head, cls_head)
        if not result:
            return result
        assert len(x) == 9 and self.nl == 4, f"Expected 9 streams, got {len(x)}"
        projection = self.ctx_o2m if cls_head is self.cv3 else self.ctx_o2o
        delta = F.pixel_shuffle(projection(x[8]), 2).flatten(2)
        original = result["scores"]
        n = delta.shape[-1]
        result["scores"] = torch.cat((original[:, :, :n] + delta, original[:, :, n:]), dim=2)
        return result


def build(parent, arm: str):
    if arm not in ARMS:
        raise ValueError(f"Unknown v3 arm: {arm}")
    assert isinstance(parent.model[-1], P2PKI2ClsDetect), (
        f"Expected HistoricalCLS P2PKI2ClsDetect, got {type(parent.model[-1])}"
    )
    model = copy.deepcopy(parent)
    if arm == "HistCLS_SchemeA_P2":
        head = model.model[-1]
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(42)
            head.__class__ = HistoricalScoreContextDetect
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

    # 1. Exact step-0 equivalence check on O2M and O2O against HistoricalCLS (non-square rect shape)
    a, b = copy.deepcopy(parent).eval(), copy.deepcopy(candidate).eval()
    x = torch.rand(1, 3, 128, 160, generator=torch.Generator().manual_seed(43))
    ra, rb = a(x)[1], b(x)[1]
    checks = {
        branch: {k: float((ra[branch][k] - rb[branch][k]).abs().max()) for k in ("boxes", "scores")}
        for branch in ("one2many", "one2one")
    }
    assert all(val == 0.0 for metrics in checks.values() for val in metrics.values()), checks

    # 2. Deployment fuse & O2M compaction parity check with NON-ZERO active weights
    c_active = copy.deepcopy(candidate).eval()
    gen = torch.Generator().manual_seed(99)
    if arm == "HistCLS_SchemeA_P2":
        c_active.model[-1].ctx_o2m.weight.normal_(0, 0.1, generator=gen)
        c_active.model[-1].ctx_o2m.bias.normal_(0, 0.1, generator=gen)
        c_active.model[-1].ctx_o2o.weight.normal_(0, 0.1, generator=gen)
        c_active.model[-1].ctx_o2o.bias.normal_(0, 0.1, generator=gen)
        # Verify P3/P4/P5 scores are untouched and boxes are untouched vs parent
        r_par = a(x)[1]["one2many"]
        r_act = c_active(x)[1]["one2many"]
        n_p2 = (128 // 4) * (160 // 4)
        p2_delta_max = float((r_act["scores"][:, :, :n_p2] - r_par["scores"][:, :, :n_p2]).abs().max())
        p345_delta_max = float((r_act["scores"][:, :, n_p2:] - r_par["scores"][:, :, n_p2:]).abs().max())
        box_delta_max = float((r_act["boxes"] - r_par["boxes"]).abs().max())
        assert p2_delta_max > 1e-3, f"Expected active P2 score delta, got {p2_delta_max}"
        assert p345_delta_max == 0.0, f"P3/P4/P5 scores altered: {p345_delta_max}"
        assert box_delta_max == 0.0, f"Boxes altered by SchemeA_P2: {box_delta_max}"
        checks["active_p2_score_delta_max"] = p2_delta_max
        checks["active_p345_score_delta_max"] = p345_delta_max
        checks["active_box_delta_max"] = box_delta_max

    from rebuilt_model import as_o2m

    unfused = copy.deepcopy(c_active)
    u_o2m = as_o2m(unfused)
    f_o2m = compact_o2m(c_active)
    yu, yf = u_o2m(x)[0], f_o2m(x)[0]
    fuse_box_err = float((yu[:, :4] - yf[:, :4]).abs().max())
    fuse_score_err = float((yu[:, 4:] - yf[:, 4:]).abs().max())
    assert fuse_box_err < 5e-3 and fuse_score_err < 1e-4, (
        f"Fuse mismatch on {arm}: box_err={fuse_box_err}, score_err={fuse_score_err}"
    )
    checks["active_fuse_box_err_px"] = fuse_box_err
    checks["active_fuse_score_err"] = fuse_score_err
    checks["deployed_o2m_params"] = sum(p.numel() for p in f_o2m.parameters())

    # 3. Gradient flow, regression isolation, and O2O detach isolation
    if arm == "HistCLS_SchemeA_P2":
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

            # 3b. Step-2 gradient into context[0] and context[2] + cv2 regression isolation
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

            # 3c. Strict O2O detach isolation (one2one must NOT backprop into context or p2_fusion)
            g3 = copy.deepcopy(candidate).train()
            g3.model[-1].ctx_o2o.weight.data.fill_(0.05)
            out3 = g3(xg)["one2one"]
            out3["scores"].sum().backward()
            ctx_o2o_leak = max(
                (float(p.grad.abs().max()) if p.grad is not None else 0.0)
                for p in list(g3.model[-1].context.parameters()) + list(g3.model[-1].p2_fusion.parameters())
            )
            assert ctx_o2o_leak == 0.0, f"O2O leaked gradient into context/p2_fusion: {ctx_o2o_leak}"
            checks["step1_ctx_grad"] = dict(o2m_w=o2m_w, o2m_b=o2m_b, o2o_w=o2o_w, o2o_b=o2o_b)
            checks["step2_context_grad"] = dict(pw=c0_g, dw=c2_g)
            checks["reg_isolation_max_cv2_grad"] = 0.0
            checks["o2o_detach_max_context_grad"] = 0.0

    return candidate, dict(
        arm=arm,
        params=n,
        added=ADDED[arm],
        inherited_tensors=len(old),
        protocol_overrides=PROTOCOL_OVERRIDES[arm],
        checks=checks,
    )
