"""V2 candidate architectures built strictly from the original October HistoricalCLS state."""
import copy
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.fusion import fuse_conv_bn_eval
from ultralytics.nn.modules import Conv
from historical_head import P2PKI2ClsDetect
from candidates import haar

ARMS = (
    "HistCLS_O2M_RegQual",
    "HistCLS_AntiAlias_L01",
    "HistCLS_Dual_L1_abs",
    "HistCLS_Dual_L13_abs",
)
ADDED = dict(
    HistCLS_O2M_RegQual=11716,
    HistCLS_AntiAlias_L01=48,
    HistCLS_Dual_L1_abs=3232,
    HistCLS_Dual_L13_abs=34384,
)


class HistoricalRegQualityDetect(P2PKI2ClsDetect):
    """Original October P2PKI2ClsDetect + O2M regression-guided spatial localization quality calibration.

    Reads detached penultimate regression features (16ch) and detached predicted box distances (4ch)
    through a lightweight 3x3 spatial Conv + zero-initialized 1x1 projection to output a class-agnostic
    localization quality residual added to O2M classification logits.
    - Preserves P2PKI2ClsDetect.forward (p2_fusion + HistoricalPKI) byte-for-byte.
    - Zero gradient flow into regression towers (cv2).
    - Exact zero-init equivalence with HistoricalCLS at step 0.
    - Zero new loss terms; trained end-to-end by native TAL IoU-weighted soft targets.
    """

    def forward_head(self, x, box_head=None, cls_head=None):
        if box_head is None or cls_head is None:
            return {}
        assert len(x) == 8 and self.nl == 4
        bs = x[0].shape[0]
        use_qual = cls_head is self.cv3
        boxes_list, scores_list = [], []
        for i in range(4):
            f_reg = box_head[i][:-1](x[i])
            b_raw = box_head[i][-1](f_reg)
            c_raw = cls_head[i](x[i + 4])
            if use_qual:
                q_in = torch.cat((f_reg.detach(), b_raw.detach()), dim=1)
                c_raw = c_raw + self.reg_qual_o2m[i](q_in)
            boxes_list.append(b_raw.view(bs, 4 * self.reg_max, -1))
            scores_list.append(c_raw.view(bs, self.nc, -1))
        return {
            "boxes": torch.cat(boxes_list, dim=-1),
            "scores": torch.cat(scores_list, dim=-1),
            "feats": x[:4],
        }


class AntiAliasConv(nn.Module):
    """Zero-initialized per-channel residual BlurPool anti-aliased strided Conv (ICML 2019).

    Shares the exact pretrained 3x3 Conv weights between the original strided path and the
    phase-aligned Conv(s=1) + 3x3 binomial BlurPool(s=2, reflect pad) path.
    - Does NOT inherit from ultralytics Conv so model.fuse() cannot corrupt it.
    - 1D per-channel alpha stays out of Muon orthogonalization.
    """

    def __init__(self, orig: nn.Module):
        super().__init__()
        self.conv = orig.conv
        self.bn = orig.bn
        self.act = orig.act
        assert self.conv.kernel_size == (3, 3) and self.conv.stride == (2, 2) and self.conv.padding == (1, 1)
        co = self.conv.out_channels
        k1 = torch.tensor([1.0, 2.0, 1.0], dtype=torch.float32) / 4.0
        k2 = torch.outer(k1, k1).view(1, 1, 3, 3).repeat(co, 1, 1, 1).contiguous()
        self.register_buffer("blur_kernel", k2)
        self.alpha = nn.Parameter(torch.zeros(co))
        self.i, self.f, self.type = orig.i, orig.f, "AntiAliasConv"
        self.np = sum(p.numel() for p in self.parameters())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w, b = self.conv.weight, self.conv.bias
        y_s2 = F.conv2d(x, w, b, stride=2, padding=1, dilation=self.conv.dilation, groups=self.conv.groups)
        y_s1 = F.conv2d(x, w, b, stride=1, padding=1, dilation=self.conv.dilation, groups=self.conv.groups)
        y_aa = F.conv2d(
            F.pad(y_s1, (1, 1, 1, 1), mode="reflect"),
            self.blur_kernel.to(dtype=y_s1.dtype),
            stride=2,
            groups=y_s1.shape[1],
        )
        mixed = y_s2 + self.alpha.to(y_s2.dtype).view(1, -1, 1, 1) * (y_aa - y_s2)
        if self.bn is not None:
            mixed = self.bn(mixed)
        return self.act(mixed)

    @torch.no_grad()
    def fuse_bn_eval(self):
        assert not self.training
        if self.bn is not None:
            self.conv = fuse_conv_bn_eval(self.conv, self.bn)
            self.bn = None
        return self


class DualBandHaarAbs(nn.Module):
    """Fixed DDLD-Dual: abs-energy pooling on signed high-frequency bands [LH, HL, HH],
    2*sigmoid zero-init gating (initial gain = 1.0), and 1D per-channel LayerScale gamma.
    """

    def __init__(self, orig: nn.Module):
        super().__init__()
        self.conv, self.bn, self.act = orig.conv, orig.bn, orig.act
        ci, co = self.conv.in_channels, self.conv.out_channels
        assert self.conv.kernel_size == (3, 3) and self.conv.stride == (2, 2) and self.conv.padding == (1, 1)
        n = 3 * ci
        mid = max(16, n // 4)
        self.gate = nn.Sequential(nn.Conv2d(n, mid, 1), nn.SiLU(), nn.Conv2d(mid, n, 1))
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)
        self.pw = nn.Conv2d(n, co, 1, bias=False)
        self.hbn = nn.BatchNorm2d(co, eps=self.bn.eps, momentum=self.bn.momentum)
        self.gamma = nn.Parameter(torch.zeros(co))
        self.i, self.f, self.type = orig.i, orig.f, "DualBandHaarAbs"
        self.np = sum(p.numel() for p in self.parameters())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bands = haar(x)
        y = torch.cat(bands[1:], 1)
        pooled = y.abs().mean((2, 3), keepdim=True)
        g = (2.0 * torch.sigmoid(self.gate(pooled).float())).to(y.dtype)
        y = y * g
        base = self.bn(self.conv(x)) if self.bn is not None else self.conv(x)
        hf = self.hbn(self.pw(y)) if self.hbn is not None else self.pw(y)
        return self.act(base + self.gamma.to(base.dtype).view(1, -1, 1, 1) * hf)

    @torch.no_grad()
    def fuse_bn_eval(self):
        assert not self.training
        if self.bn is not None:
            self.conv = fuse_conv_bn_eval(self.conv, self.bn)
            self.bn = None
        if self.hbn is not None:
            self.pw = fuse_conv_bn_eval(self.pw, self.hbn)
            self.hbn = None
        return self


def build(parent, arm: str):
    if arm not in ARMS:
        raise ValueError(f"Unknown v2 arm: {arm}")
    assert isinstance(parent.model[-1], P2PKI2ClsDetect), f"Expected HistoricalCLS P2PKI2ClsDetect, got {type(parent.model[-1])}"
    model = copy.deepcopy(parent)
    head = model.model[-1]
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        if arm == "HistCLS_O2M_RegQual":
            head.__class__ = HistoricalRegQualityDetect
            modules = []
            for i in range(4):
                c = Conv(20, 16, 3)
                c.bn.eps = head.cv2[i][0].bn.eps
                c.bn.momentum = head.cv2[i][0].bn.momentum
                proj = nn.Conv2d(16, 1, 1)
                nn.init.zeros_(proj.weight)
                nn.init.zeros_(proj.bias)
                modules.append(nn.Sequential(c, proj))
            head.reg_qual_o2m = nn.ModuleList(modules)
        elif arm == "HistCLS_AntiAlias_L01":
            model.model[0] = AntiAliasConv(model.model[0])
            model.model[1] = AntiAliasConv(model.model[1])
        elif arm == "HistCLS_Dual_L1_abs":
            model.model[1] = DualBandHaarAbs(model.model[1])
        elif arm == "HistCLS_Dual_L13_abs":
            model.model[1] = DualBandHaarAbs(model.model[1])
            model.model[3] = DualBandHaarAbs(model.model[3])
    return model


@torch.no_grad()
def prepare(parent, arm: str):
    candidate = build(parent, arm)
    old = parent.state_dict()
    new = candidate.state_dict()
    assert all(k in new and torch.equal(v, new[k]) for k, v in old.items()), "HistoricalCLS parent weight inheritance broken!"
    n = sum(p.numel() for p in candidate.parameters())
    assert n == 3195674 + ADDED[arm], f"Param mismatch: got {n}, expected {3195674 + ADDED[arm]}"

    # 1. Exact step-0 equivalence check on O2M and O2O against HistoricalCLS
    a, b = copy.deepcopy(parent).eval(), copy.deepcopy(candidate).eval()
    x = torch.randn(1, 3, 128, 160, generator=torch.Generator().manual_seed(43))
    ra, rb = a(x)[1], b(x)[1]
    checks = {
        branch: {k: float((ra[branch][k] - rb[branch][k]).abs().max()) for k in ("boxes", "scores")}
        for branch in ("one2many", "one2one")
    }
    assert all(val == 0.0 for metrics in checks.values() for val in metrics.values()), checks

    # 2. Deployment fuse safety check with NON-ZERO active weights
    c_active = copy.deepcopy(candidate).eval()
    gen = torch.Generator().manual_seed(99)
    if arm == "HistCLS_O2M_RegQual":
        for m in c_active.model[-1].reg_qual_o2m:
            m[-1].weight.normal_(0, 0.1, generator=gen)
            m[-1].bias.normal_(0, 0.1, generator=gen)
    elif arm == "HistCLS_AntiAlias_L01":
        c_active.model[0].alpha.normal_(0.5, 0.2, generator=gen)
        c_active.model[1].alpha.normal_(0.5, 0.2, generator=gen)
    elif arm in ("HistCLS_Dual_L1_abs", "HistCLS_Dual_L13_abs"):
        for idx in ([1] if arm == "HistCLS_Dual_L1_abs" else [1, 3]):
            c_active.model[idx].gamma.normal_(0.2, 0.1, generator=gen)
            c_active.model[idx].gate[-1].weight.normal_(0, 0.1, generator=gen)
    unfused = copy.deepcopy(c_active)
    from rebuilt_model import as_o2m
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

    # 3. Gradient flow & isolation check
    with torch.enable_grad():
        g_mod = copy.deepcopy(candidate).train()
        xg = torch.randn(2, 3, 128, 128, generator=torch.Generator().manual_seed(7))
        out1 = g_mod(xg)["one2many"]
        loss1 = (out1["scores"] ** 2).mean() + (out1["boxes"] ** 2).mean()
        loss1.backward()
        if arm == "HistCLS_O2M_RegQual":
            s1_norms = [float(m[-1].weight.grad.abs().max()) for m in g_mod.model[-1].reg_qual_o2m]
            assert all(v > 0 for v in s1_norms), f"Dead O2M_RegQual step-1 grad: {s1_norms}"
            g_iso = copy.deepcopy(candidate).train()
            for m in g_iso.model[-1].reg_qual_o2m:
                m[-1].weight.data.fill_(0.1)
            o_iso = g_iso(xg)["one2many"]
            o_iso["scores"].sum().backward()
            cv2_grads = [float(p.grad.abs().max()) if p.grad is not None else 0.0 for p in g_iso.model[-1].cv2.parameters()]
            assert max(cv2_grads) == 0.0, f"Gradient leaked from O2M_RegQual into cv2: {cv2_grads}"
            checks["reg_isolation_max_cv2_grad"] = 0.0
            checks["step1_proj_max_grad"] = s1_norms
        elif arm == "HistCLS_AntiAlias_L01":
            a0 = float(g_mod.model[0].alpha.grad.abs().max())
            a1 = float(g_mod.model[1].alpha.grad.abs().max())
            assert a0 > 0 and a1 > 0, f"Dead AntiAlias_L01 alpha grad: {a0}, {a1}"
            checks["step1_alpha_max_grad"] = [a0, a1]
        elif arm in ("HistCLS_Dual_L1_abs", "HistCLS_Dual_L13_abs"):
            idxs = [1] if arm == "HistCLS_Dual_L1_abs" else [1, 3]
            gnorms = [float(g_mod.model[i].gamma.grad.abs().max()) for i in idxs]
            assert all(v > 0 for v in gnorms), f"Dead Dual gamma grad: {gnorms}"
            checks["step1_gamma_max_grad"] = gnorms

    return candidate, dict(arm=arm, params=n, added=ADDED[arm], inherited_tensors=len(old), checks=checks)


def compact_o2m(model):
    from rebuilt_model import as_o2m
    out = as_o2m(model)
    out.fuse(verbose=False)
    for m in out.modules():
        if isinstance(m, (AntiAliasConv, DualBandHaarAbs)):
            m.fuse_bn_eval()
    return out
