"""V4 candidate architectures built strictly from the original October HistoricalCLS state.
Arms:
  1) HistoricalCLS_seed100      : Pure original October HistoricalCLS, seed=100 (+0 params)
  2) HistCLS_SchemeA_P2_seed100 : Original HistoricalCLS + Scheme A P2 context, seed=100 (+12,376 params)
  3) HistCLS_SchemeA_P2P3       : Original HistoricalCLS + Scheme A P2 & P3 context, seed=42 (+26,800 params)
"""
import copy
import torch
from torch import nn
import torch.nn.functional as F
from ultralytics.nn.modules.head import Detect
from historical_head import P2PKI2ClsDetect
from candidates_v3 import HistoricalScoreContextDetect

ARMS = (
    "HistoricalCLS_seed100",
    "HistCLS_SchemeA_P2_seed100",
    "HistCLS_SchemeA_P2P3",
)
ADDED = dict(
    HistoricalCLS_seed100=0,
    HistCLS_SchemeA_P2_seed100=12376,
    HistCLS_SchemeA_P2P3=26800,
)
SEEDS = dict(
    HistoricalCLS_seed100=100,
    HistCLS_SchemeA_P2_seed100=100,
    HistCLS_SchemeA_P2P3=42,
)


class HistoricalScoreContextP2P3Detect(P2PKI2ClsDetect):
    """Original October P2PKI2ClsDetect + symmetric P2 and P3 classification score context.

    - P2 residual: [pixel_unshuffle(p2, 2), p3] (320 -> 32 @ stride 8) -> zero-init pixel_shuffle -> P2 cls logits.
    - P3 residual: [pixel_unshuffle(p3, 2), p4] (384 -> 32 @ stride 16) -> zero-init pixel_shuffle -> P3 cls logits.
    - Preserves P2PKI2ClsDetect p2_fusion (BN eps=0.001, momentum=0.03) and HistoricalPKI (P4/P5) verbatim.
    - Zero change to regression towers (cv2 / one2one_cv2) or P4/P5 classification logits.
    - Detect.forward detaches all 10 streams before one2one, preserving strict O2O gradient isolation.
    - Exact 0.0 step-0 equivalence with original October HistoricalCLS and HistCLS_SchemeA_P2.
    """

    def forward(self, features):
        c2, p3, p4, p5 = features
        context_pki = self.paper2_pki2([p3, p4, p5])
        p2 = self.p2_fusion(torch.cat((c2, F.interpolate(p3, size=c2.shape[-2:], mode="nearest")), 1))
        p2_ctx = self.context(torch.cat((F.pixel_unshuffle(p2, 2), p3), 1))
        p3_ctx = self.context_p3(torch.cat((F.pixel_unshuffle(p3, 2), p4), 1))
        boxes = [p2, p3, p4, p5]
        classes = [p2, *context_pki]
        return Detect.forward(self, boxes + classes + [p2_ctx, p3_ctx])

    def forward_head(self, x, box_head=None, cls_head=None):
        result = super().forward_head(x[:8], box_head, cls_head)
        if not result:
            return result
        assert len(x) == 10 and self.nl == 4, f"Expected 10 streams, got {len(x)}"
        is_o2m = cls_head is self.cv3
        proj_p2 = self.ctx_o2m if is_o2m else self.ctx_o2o
        proj_p3 = self.ctx_p3_o2m if is_o2m else self.ctx_p3_o2o
        delta_p2 = F.pixel_shuffle(proj_p2(x[8]), 2).flatten(2)
        delta_p3 = F.pixel_shuffle(proj_p3(x[9]), 2).flatten(2)
        original = result["scores"]
        n2 = delta_p2.shape[-1]
        n3 = delta_p3.shape[-1]
        assert n2 == x[0].shape[-2] * x[0].shape[-1], (n2, x[0].shape)
        assert n3 == x[1].shape[-2] * x[1].shape[-1], (n3, x[1].shape)
        result["scores"] = torch.cat(
            (
                original[:, :, :n2] + delta_p2,
                original[:, :, n2 : n2 + n3] + delta_p3,
                original[:, :, n2 + n3 :],
            ),
            dim=2,
        )
        return result


def build(parent, arm: str):
    if arm not in ARMS:
        raise ValueError(f"Unknown v4 arm: {arm}")
    assert isinstance(parent.model[-1], P2PKI2ClsDetect), (
        f"Expected HistoricalCLS P2PKI2ClsDetect, got {type(parent.model[-1])}"
    )
    model = copy.deepcopy(parent)
    head = model.model[-1]
    init_seed = SEEDS[arm]

    if arm == "HistoricalCLS_seed100":
        return model

    if arm == "HistCLS_SchemeA_P2_seed100":
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(init_seed)
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

    if arm == "HistCLS_SchemeA_P2P3":
        # Initialize P2 context with seed=42 first so its initial state matches HistCLS_SchemeA_P2 (seed=42) identically,
        # then initialize the symmetric P3 context branch.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(42)
            head.__class__ = HistoricalScoreContextP2P3Detect
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

            head.context_p3 = nn.Sequential(
                nn.Conv2d(384, 32, 1, bias=False),
                nn.SiLU(),
                nn.Conv2d(32, 32, 3, padding=1, groups=32, bias=False),
                nn.SiLU(),
            )
            head.ctx_p3_o2m = nn.Conv2d(32, 28, 1)
            head.ctx_p3_o2o = nn.Conv2d(32, 28, 1)
            for m in (head.ctx_p3_o2m, head.ctx_p3_o2o):
                nn.init.zeros_(m.weight)
                nn.init.zeros_(m.bias)
        return model

    raise AssertionError(arm)


def compact_o2m(model):
    from rebuilt_model import as_o2m

    out = as_o2m(model)
    head = out.model[-1]
    if isinstance(head, (HistoricalScoreContextDetect, HistoricalScoreContextP2P3Detect)):
        if hasattr(head, "ctx_o2o"):
            del head.ctx_o2o
        if hasattr(head, "ctx_p3_o2o"):
            del head.ctx_p3_o2o
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

    # 1. Step-0 exact equivalence on both standard non-square (128x160) AND odd-32-multiple rect (416x640)
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

    # 2. Active non-zero isolation & deployment fuse parity check
    x = torch.rand(1, 3, 128, 160, generator=torch.Generator().manual_seed(43))
    c_active = copy.deepcopy(candidate).eval()
    gen = torch.Generator().manual_seed(99)
    n_p2 = (128 // 4) * (160 // 4)
    n_p3 = (128 // 8) * (160 // 8)

    if arm in ("HistCLS_SchemeA_P2_seed100", "HistCLS_SchemeA_P2P3"):
        c_active.model[-1].ctx_o2m.weight.normal_(0, 0.1, generator=gen)
        c_active.model[-1].ctx_o2m.bias.normal_(0, 0.1, generator=gen)
        c_active.model[-1].ctx_o2o.weight.normal_(0, 0.1, generator=gen)
        c_active.model[-1].ctx_o2o.bias.normal_(0, 0.1, generator=gen)
    if arm == "HistCLS_SchemeA_P2P3":
        c_active.model[-1].ctx_p3_o2m.weight.normal_(0, 0.1, generator=gen)
        c_active.model[-1].ctx_p3_o2m.bias.normal_(0, 0.1, generator=gen)
        c_active.model[-1].ctx_p3_o2o.weight.normal_(0, 0.1, generator=gen)
        c_active.model[-1].ctx_p3_o2o.bias.normal_(0, 0.1, generator=gen)

    if arm != "HistoricalCLS_seed100":
        r_par = a(x)[1]["one2many"]
        r_act = c_active(x)[1]["one2many"]
        p2_diff = float((r_act["scores"][:, :, :n_p2] - r_par["scores"][:, :, :n_p2]).abs().max())
        p3_diff = float((r_act["scores"][:, :, n_p2 : n_p2 + n_p3] - r_par["scores"][:, :, n_p2 : n_p2 + n_p3]).abs().max())
        p45_diff = float((r_act["scores"][:, :, n_p2 + n_p3 :] - r_par["scores"][:, :, n_p2 + n_p3 :]).abs().max())
        box_diff = float((r_act["boxes"] - r_par["boxes"]).abs().max())
        assert p2_diff > 1e-3, p2_diff
        if arm == "HistCLS_SchemeA_P2_seed100":
            assert p3_diff == 0.0 and p45_diff == 0.0, (p3_diff, p45_diff)
        else:
            assert p3_diff > 1e-3 and p45_diff == 0.0, (p3_diff, p45_diff)
        assert box_diff == 0.0, f"Boxes altered: {box_diff}"
        checks["active_isolation"] = dict(p2_diff=p2_diff, p3_diff=p3_diff, p45_diff=p45_diff, box_diff=box_diff)

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

    # 3. Gradient flow, cv2 isolation, PKI isolation, and O2O detach isolation
    if arm != "HistoricalCLS_seed100":
        with torch.enable_grad():
            xg = torch.rand(2, 3, 128, 128, generator=torch.Generator().manual_seed(7))
            # 3a. Step-1 gradient into zero-init projections
            g1 = copy.deepcopy(candidate).train()
            out1 = g1(xg)
            loss1 = (out1["one2many"]["scores"] ** 2).mean() + (out1["one2one"]["scores"] ** 2).mean()
            loss1.backward()
            proj_names = ["ctx_o2m", "ctx_o2o"] + (["ctx_p3_o2m", "ctx_p3_o2o"] if arm == "HistCLS_SchemeA_P2P3" else [])
            s1 = {}
            for pname in proj_names:
                mod = getattr(g1.model[-1], pname)
                wg = float(mod.weight.grad.abs().max())
                bg = float(mod.bias.grad.abs().max())
                assert wg > 0.0 and bg > 0.0, (pname, wg, bg)
                s1[pname] = dict(w=wg, b=bg)

            # 3b. Step-2 gradient into context & context_p3 + cv2 isolation
            g2 = copy.deepcopy(candidate).train()
            g2.model[-1].ctx_o2m.weight.data.fill_(0.05)
            if arm == "HistCLS_SchemeA_P2P3":
                g2.model[-1].ctx_p3_o2m.weight.data.fill_(0.05)
            out2 = g2(xg)["one2many"]
            out2["scores"].sum().backward()
            s2 = dict(
                p2_pw=float(g2.model[-1].context[0].weight.grad.abs().max()),
                p2_dw=float(g2.model[-1].context[2].weight.grad.abs().max()),
            )
            assert s2["p2_pw"] > 0.0 and s2["p2_dw"] > 0.0, s2
            if arm == "HistCLS_SchemeA_P2P3":
                s2["p3_pw"] = float(g2.model[-1].context_p3[0].weight.grad.abs().max())
                s2["p3_dw"] = float(g2.model[-1].context_p3[2].weight.grad.abs().max())
                assert s2["p3_pw"] > 0.0 and s2["p3_dw"] > 0.0, s2
            cv2_max = max(
                (float(p.grad.abs().max()) if p.grad is not None else 0.0)
                for p in g2.model[-1].cv2.parameters()
            )
            assert cv2_max == 0.0, f"Gradient leaked into cv2: {cv2_max}"

            # 3c. Strict O2O detach isolation
            g3 = copy.deepcopy(candidate).train()
            g3.model[-1].ctx_o2o.weight.data.fill_(0.05)
            if arm == "HistCLS_SchemeA_P2P3":
                g3.model[-1].ctx_p3_o2o.weight.data.fill_(0.05)
            out3 = g3(xg)["one2one"]
            out3["scores"].sum().backward()
            guarded = list(g3.model[-1].context.parameters()) + list(g3.model[-1].p2_fusion.parameters())
            if arm == "HistCLS_SchemeA_P2P3":
                guarded += list(g3.model[-1].context_p3.parameters())
            leak = max((float(p.grad.abs().max()) if p.grad is not None else 0.0) for p in guarded)
            assert leak == 0.0, f"O2O leaked gradient into shared context/p2_fusion: {leak}"

            checks["step1_proj_grad"] = s1
            checks["step2_context_grad"] = s2
            checks["reg_isolation_max_cv2_grad"] = 0.0
            checks["o2o_detach_max_shared_grad"] = 0.0

    return candidate, dict(
        arm=arm,
        seed=SEEDS[arm],
        params=n,
        added=ADDED[arm],
        inherited_tensors=len(old),
        checks=checks,
    )
