"""V6 candidates & warp transforms built strictly from CLS-P2A (October HistoricalCLS + SchemeA_P2).

Arms:
  1) b1      : Backbone L0->L2 SPD + DEConv zero-init shallow high-frequency side-branch (+38,080 over CLS-P2A)
  2) warp50  : Legacy alias for CLS-P2A + #5 (kept paused in runs/warp50)
  3) p2a_w50 : CLS-P2A + #5 single-breakpoint 50% road vertical resampling (256/384, exact inverse-y eval)
  4) b1_w50  : Factorial combination (b1 L0->L2 SPD+DEConv side-branch + warp50 256/384 road vertical resampling)
"""
import copy
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from ultralytics.data.dataset import YOLODataset
from ultralytics.nn.modules import C3k2, Conv
from ultralytics.nn.tasks import RepVGGDW
from ultralytics.utils import ops
from ultralytics.utils.torch_utils import fuse_conv_and_bn

import candidates_v3
import candidates_v5

ARMS = (
    "b1",
    "warp50",
    "p2a_w50",
    "b1_w50",
)
SEEDS = dict(
    b1=42,
    warp50=42,
    p2a_w50=42,
    b1_w50=42,
)
# Added over CLS-P2A (3,208,050 params); total added over HistoricalCLS (3,195,674) is 12,376 + ADDED_OVER_P2A.
ADDED_OVER_P2A = dict(
    b1=38080,
    warp50=0,
    p2a_w50=0,
    b1_w50=38080,
)
ADDED_OVER_HIST = dict(
    b1=12376 + 38080,
    warp50=12376,
    p2a_w50=12376,
    b1_w50=12376 + 38080,
)
PROTOCOL_OVERRIDES = dict(
    b1={},
    warp50={},
    p2a_w50={},
    b1_w50={},
)


def has_b1(arm: str) -> bool:
    return arm in ("b1", "b1_w50")


def has_warp50(arm: str) -> bool:
    return arm in ("warp50", "p2a_w50", "b1_w50")
WARP50_SPLIT_Y = 0.5
WARP50_TOP_FRAC = 256.0 / 640.0  # 0.4 -> upper 256 rows (0.8x), lower 384 rows (1.2x) at imgsz=640


# ---------------------------------------------------------------------------
# Arm 1 (B1): L0 -> SPD(2x) -> 1x1 Reduce -> DEConv2d+BN+SiLU -> ZeroInit 1x1 -> +L2
# ---------------------------------------------------------------------------
class DEConv2dBNAct(RepVGGDW):
    """5-branch Detail-Enhanced Conv (VC + CDC + ADC + HDC + VDC) + BN + SiLU.

    Subclasses RepVGGDW so Ultralytics BaseModel.fuse() calls .fuse() automatically
    and rewires .forward = .forward_fuse into a single 3x3 Conv2d + bias + SiLU.
    """

    def __init__(self, c: int = 32):
        nn.Module.__init__(self)
        self.c = int(c)
        self.dim = self.c
        self.vc = nn.Conv2d(self.c, self.c, 3, padding=1, bias=False)
        self.cdc = nn.Conv2d(self.c, self.c, 3, padding=1, bias=False)
        self.adc = nn.Conv2d(self.c, self.c, 3, padding=1, bias=False)
        self.hdc = nn.Conv2d(self.c, self.c, (3, 1), bias=False)
        self.vdc = nn.Conv2d(self.c, self.c, (1, 3), bias=False)

        scale = 1.0 / math.sqrt(5.0)
        for m in (self.vc, self.cdc, self.adc, self.hdc, self.vdc):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            m.weight.data.mul_(scale)

        center_mask = torch.zeros(1, 1, 3, 3)
        center_mask[0, 0, 1, 1] = 1.0
        self.register_buffer("_center_mask", center_mask, persistent=False)
        # Clockwise angular permutation on the 3x3 outer ring: 0->1->2->5->8->7->6->3->0, center 4->4.
        self.register_buffer("_adc_perm", torch.tensor([1, 2, 5, 0, 4, 8, 3, 6, 7], dtype=torch.long), persistent=False)

        self.bn = nn.BatchNorm2d(self.c, eps=0.001, momentum=0.03)
        self.act = nn.SiLU(inplace=True)

    def _cdc_weight(self) -> torch.Tensor:
        w = self.cdc.weight
        mask = self._center_mask.to(device=w.device, dtype=w.dtype)
        return w - w.sum(dim=(2, 3), keepdim=True) * mask

    def _adc_weight(self) -> torch.Tensor:
        w = self.adc.weight
        co, ci = w.shape[:2]
        perm = self._adc_perm.to(device=w.device)
        w_rot = w.reshape(co, ci, 9).index_select(2, perm).reshape_as(w)
        return w - w_rot

    def _hdc_weight(self) -> torch.Tensor:
        w = self.hdc.weight
        z = torch.zeros_like(w)
        return torch.cat((w, z, -w), dim=3)

    def _vdc_weight(self) -> torch.Tensor:
        w = self.vdc.weight
        z = torch.zeros_like(w)
        return torch.cat((w, z, -w), dim=2)

    def _equivalent_weight(self) -> torch.Tensor:
        return self.vc.weight + self._cdc_weight() + self._adc_weight() + self._hdc_weight() + self._vdc_weight()

    def forward_explicit_branches(self, x: torch.Tensor) -> torch.Tensor:
        """Unmerged reference forward summing all 5 differential branches explicitly."""
        y = (
            self.vc(x)
            + F.conv2d(x, self._cdc_weight(), bias=None, stride=1, padding=1)
            + F.conv2d(x, self._adc_weight(), bias=None, stride=1, padding=1)
            + F.conv2d(x, self._hdc_weight(), bias=None, stride=1, padding=1)
            + F.conv2d(x, self._vdc_weight(), bias=None, stride=1, padding=1)
        )
        return self.act(self.bn(y))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self, "conv") and not hasattr(self, "bn"):
            return self.forward_fuse(x)
        w_eq = self._equivalent_weight()
        return self.act(self.bn(F.conv2d(x, w_eq, bias=None, stride=1, padding=1)))

    def forward_fuse(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv(x))

    @torch.no_grad()
    def fuse(self):
        if not hasattr(self, "bn"):
            return
        w_eq = self._equivalent_weight().detach()
        equiv = nn.Conv2d(self.c, self.c, 3, padding=1, bias=False).to(device=w_eq.device, dtype=w_eq.dtype)
        equiv.weight.data.copy_(w_eq)
        self.conv = fuse_conv_and_bn(equiv, self.bn)
        for attr in ("vc", "cdc", "adc", "hdc", "vdc", "bn"):
            if hasattr(self, attr):
                delattr(self, attr)


class ShallowSPDDEConvSide(nn.Module):
    """L0 (B, 16, H/2, W/2) -> PixelUnshuffle(2) -> Conv1x1 -> DEConv3x3 -> ZeroInit Conv1x1 -> (B, 64, H/4, W/4)."""

    def __init__(self, in_c: int = 16, mid_c: int = 32, out_c: int = 64):
        super().__init__()
        self.reduce = Conv(in_c * 4, mid_c, k=1, s=1)
        self.reduce.bn = nn.BatchNorm2d(mid_c, eps=0.001, momentum=0.03)
        self.deconv = DEConv2dBNAct(mid_c)
        self.proj = nn.Conv2d(mid_c, out_c, 1, bias=True)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, l0: torch.Tensor) -> torch.Tensor:
        return self.proj(self.deconv(self.reduce(F.pixel_unshuffle(l0, 2))))


class ShallowSideC3k2(C3k2):
    """L2 C3k2 wrapper adding the L0 shallow high-frequency residual without shifting model indices 3..23."""

    def forward(self, x):
        if isinstance(x, (list, tuple)):
            l0, l1 = x
            return super().forward(l1) + self.shallow_side(l0)
        raise RuntimeError("ShallowSideC3k2 expects [l0, l1] from model.save=[0, ...] with f=[0, -1].")


def build(parent, arm: str):
    if arm not in ARMS:
        raise ValueError(f"Unknown v6 arm: {arm}")
    model = candidates_v5.build(parent, "p2_cm0")  # Strict CLS-P2A base (HistoricalCLS + SchemeA_P2 seed42)
    if has_b1(arm):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(SEEDS[arm])
            l2 = model.model[2]
            assert isinstance(l2, C3k2) and l2.i == 2 and l2.f == -1
            l0 = model.model[0]
            assert l0.conv.out_channels == 16 and l2.cv2.conv.out_channels == 64
            l2.__class__ = ShallowSideC3k2
            l2.f = [0, -1]
            l2.shallow_side = ShallowSPDDEConvSide(in_c=16, mid_c=32, out_c=64)
            l2.type = "ShallowSideC3k2"
            l2.np = sum(p.numel() for p in l2.parameters())
            model.save = sorted(set(model.save) | {0})
    return model


def compact_o2m(model):
    out = candidates_v5.compact_o2m(model)
    return out


# ---------------------------------------------------------------------------
# Arm 2 (#5 / warp50): Single-Breakpoint Vertical Resampling + Inverse-y Eval
# ---------------------------------------------------------------------------
def warp_y_forward_np(y: np.ndarray, split_y: float = WARP50_SPLIT_Y, top_frac: float = WARP50_TOP_FRAC) -> np.ndarray:
    """Piecewise-linear forward normalized y mapping (upper compressed, lower road expanded)."""
    y = np.asarray(y, dtype=np.float64)
    s_top = top_frac / split_y
    s_bot = (1.0 - top_frac) / (1.0 - split_y)
    return np.where(y <= split_y, y * s_top, top_frac + (y - split_y) * s_bot)


def warp_y_inverse_np(yw: np.ndarray, split_y: float = WARP50_SPLIT_Y, top_frac: float = WARP50_TOP_FRAC) -> np.ndarray:
    """Exact inverse piecewise-linear normalized y mapping (extends linearly outside [0, 1])."""
    yw = np.asarray(yw, dtype=np.float64)
    inv_top = split_y / top_frac
    inv_bot = (1.0 - split_y) / (1.0 - top_frac)
    return np.where(yw <= top_frac, yw * inv_top, split_y + (yw - top_frac) * inv_bot)


def warp_y_inverse_torch(yw: torch.Tensor, split_y: float = WARP50_SPLIT_Y, top_frac: float = WARP50_TOP_FRAC) -> torch.Tensor:
    """Differentiable/GPU exact inverse piecewise-linear normalized y mapping (unclamped before NMS)."""
    inv_top = split_y / top_frac
    inv_bot = (1.0 - split_y) / (1.0 - top_frac)
    return torch.where(yw <= top_frac, yw * inv_top, split_y + (yw - top_frac) * inv_bot)


def effective_warp_frac(out_size: int, split_y: float = WARP50_SPLIT_Y, top_frac: float = WARP50_TOP_FRAC) -> tuple[int, int, float]:
    """Integer output row split (top_h, bot_h) and exact effective top_frac = top_h / out_size."""
    out_i = int(out_size)
    top_h = int(round(out_i * float(top_frac)))
    bot_h = out_i - top_h
    assert top_h > 0 and bot_h > 0, (out_size, split_y, top_frac, top_h, bot_h)
    return top_h, bot_h, float(top_h) / float(out_i)


def warp_image(im: np.ndarray, out_size: int, split_y: float = WARP50_SPLIT_Y, top_frac: float = WARP50_TOP_FRAC) -> np.ndarray:
    """Resample native image directly to square out_size x out_size using split cv2.resize (0 pixel diff on 320/320 identity)."""
    h0 = im.shape[0]
    split_px = int(round(h0 * float(split_y)))
    top_h, bot_h, _ = effective_warp_frac(out_size, split_y, top_frac)
    top = cv2.resize(im[:split_px], (int(out_size), top_h), interpolation=cv2.INTER_LINEAR)
    bot = cv2.resize(im[split_px:], (int(out_size), bot_h), interpolation=cv2.INTER_LINEAR)
    warped = np.vstack((top, bot))
    if warped.ndim == 2:
        warped = warped[..., None]
    return warped


def warp_bboxes_xywh_np(
    bboxes: np.ndarray, split_y: float = WARP50_SPLIT_Y, top_frac: float = WARP50_TOP_FRAC
) -> np.ndarray:
    """Warp normalized xywh boxes [x, y, w, h] forward under the piecewise-linear y transform."""
    if len(bboxes) == 0:
        return bboxes
    out = np.asarray(bboxes, dtype=np.float32).copy()
    y1 = out[:, 1] - out[:, 3] * 0.5
    y2 = out[:, 1] + out[:, 3] * 0.5
    y1w = warp_y_forward_np(y1, split_y=split_y, top_frac=top_frac)
    y2w = warp_y_forward_np(y2, split_y=split_y, top_frac=top_frac)
    out[:, 1] = ((y1w + y2w) * 0.5).astype(np.float32)
    out[:, 3] = np.maximum(y2w - y1w, 1e-6).astype(np.float32)
    return out


class Warp50YOLODataset(YOLODataset):
    """Pre-Mosaic native-1024 -> imgsz single-breakpoint road vertical resampling dataset."""

    def __init__(
        self,
        *args,
        warp_split_y: float = WARP50_SPLIT_Y,
        warp_top_frac: float = WARP50_TOP_FRAC,
        **kwargs,
    ):
        self.warp_split_y = float(warp_split_y)
        self.warp_top_frac = float(warp_top_frac)
        super().__init__(*args, **kwargs)

    def load_image(self, i: int, rect_mode: bool = True) -> tuple[np.ndarray, tuple[int, int], tuple[int, int]]:
        im, f = self.ims[i], self.im_files[i]
        if im is None:
            im = self.read_image(f)
            if im is None:
                raise FileNotFoundError(f"Image Not Found {f}")
            h0, w0 = im.shape[:2]
            im = warp_image(im, self.imgsz, split_y=self.warp_split_y, top_frac=self.warp_top_frac)
            if im.ndim == 2:
                im = im[..., None]
            if self.augment:
                self.ims[i], self.im_hw0[i], self.im_hw[i] = im, (h0, w0), im.shape[:2]
                self.buffer.append(i)
                if 1 < len(self.buffer) >= self.max_buffer_length:
                    j = self.buffer.pop(0)
                    if self.cache != "ram":
                        self.ims[j], self.im_hw0[j], self.im_hw[j] = None, None, None
            return im, (h0, w0), im.shape[:2]
        return self.ims[i], self.im_hw0[i], self.im_hw[i]

    def get_image_and_label(self, index: int) -> dict[str, Any]:
        label = copy.deepcopy(self.labels[index])
        label.pop("shape", None)
        label["img"], label["ori_shape"], label["resized_shape"] = self.load_image(index)
        label["ratio_pad"] = (
            label["resized_shape"][0] / label["ori_shape"][0],
            label["resized_shape"][1] / label["ori_shape"][1],
        )
        if self.rect:
            label["rect_shape"] = self.batch_shapes[self.batch[index]]
        if label.get("normalized", True) and label.get("bbox_format", "xywh") == "xywh" and len(label["bboxes"]):
            ori_h = float(label["ori_shape"][0])
            eff_split_y = float(int(round(ori_h * self.warp_split_y))) / ori_h
            _, _, eff_top_frac = effective_warp_frac(self.imgsz, eff_split_y, self.warp_top_frac)
            label["bboxes"] = warp_bboxes_xywh_np(
                label["bboxes"], split_y=eff_split_y, top_frac=eff_top_frac
            )
        return self.update_labels_info(label)


def _unpack_ori_and_ratio_pad(batch_ori_shape, batch_ratio_pad):
    ori_h = float(batch_ori_shape[0])
    ori_w = float(batch_ori_shape[1])
    gain_h = float(batch_ratio_pad[0][0])
    gain_w = float(batch_ratio_pad[0][1])
    pad_x = float(batch_ratio_pad[1][0])
    pad_y = float(batch_ratio_pad[1][1])
    return ori_h, ori_w, gain_h, gain_w, pad_x, pad_y


class InverseWarpValidatorMixin:
    """Inverse-maps predicted boxes from warped input space back to original 1024x1024 coords BEFORE NMS."""

    warp_split_y: float = WARP50_SPLIT_Y
    warp_top_frac: float = WARP50_TOP_FRAC

    def build_dataset(self, img_path: str, mode: str = "val", batch: int | None = None):
        return Warp50YOLODataset(
            img_path=img_path,
            imgsz=self.args.imgsz,
            batch_size=batch,
            augment=False,
            hyp=self.args,
            rect=self.args.rect,
            cache=None,
            single_cls=self.args.single_cls or False,
            stride=int(self.stride),
            pad=0.5,
            prefix=f"{mode}: ",
            task=self.args.task,
            classes=self.args.classes,
            data=self.data,
            fraction=1.0,
            warp_split_y=self.warp_split_y,
            warp_top_frac=self.warp_top_frac,
        )

    def preprocess(self, batch: dict[str, Any]) -> dict[str, Any]:
        batch = super().preprocess(batch)
        self._cur_warp_batch = batch
        return batch

    def _inverse_xyxy_to_orig(self, xyxy: torch.Tensor, batch_ori_shape, batch_ratio_pad) -> torch.Tensor:
        if xyxy.numel() == 0:
            return xyxy
        ori_h, _ori_w, gain_h, gain_w, pad_x, pad_y = _unpack_ori_and_ratio_pad(batch_ori_shape, batch_ratio_pad)
        h_warp = ori_h * gain_h
        eff_split_y = float(int(round(ori_h * self.warp_split_y))) / ori_h
        _, _, eff_top_frac = effective_warp_frac(int(round(h_warp)), eff_split_y, self.warp_top_frac)
        out = xyxy.clone()
        out[:, 0] = (out[:, 0] - pad_x) / gain_w
        out[:, 2] = (out[:, 2] - pad_x) / gain_w
        y1_wn = (out[:, 1] - pad_y) / h_warp
        y2_wn = (out[:, 3] - pad_y) / h_warp
        out[:, 1] = warp_y_inverse_torch(y1_wn, eff_split_y, eff_top_frac) * ori_h
        out[:, 3] = warp_y_inverse_torch(y2_wn, eff_split_y, eff_top_frac) * ori_h
        return out

    def postprocess(self, preds):
        if self.training and getattr(self, "box_head", None) is not None and self.box_head.end2end:
            decoded = self.box_head._inference(preds[1]["one2many"])
            boxes = ops.xyxy2xywh(decoded[:, :4].transpose(1, 2)).transpose(1, 2)
            raw_pred = torch.cat([boxes, decoded[:, 4:]], dim=1)
        elif isinstance(preds, (list, tuple)):
            raw_pred = preds[0]
        else:
            raw_pred = preds

        batch = getattr(self, "_cur_warp_batch", None)
        if batch is not None:
            raw_pred = raw_pred.clone()
            bs = raw_pred.shape[0]
            for si in range(bs):
                xywh_si = raw_pred[si, :4, :].transpose(0, 1)  # (N, 4)
                xyxy_si = ops.xywh2xyxy(xywh_si)
                xyxy_orig = self._inverse_xyxy_to_orig(xyxy_si, batch["ori_shape"][si], batch["ratio_pad"][si])
                raw_pred[si, :4, :] = ops.xyxy2xywh(xyxy_orig).transpose(0, 1)

        from ultralytics.models.yolo.detect.val import DetectionValidator

        return DetectionValidator.postprocess(self, raw_pred)

    def scale_preds(self, predn: dict[str, torch.Tensor], pbatch: dict[str, Any]) -> dict[str, torch.Tensor]:
        return {
            **predn,
            "bboxes": ops.clip_boxes(predn["bboxes"].clone(), pbatch["ori_shape"]),
        }

    def _prepare_batch(self, si: int, batch: dict[str, Any]) -> dict[str, Any]:
        pbatch = super()._prepare_batch(si, batch)
        if pbatch["bboxes"].numel() > 0:
            orig_boxes = self._inverse_xyxy_to_orig(pbatch["bboxes"], pbatch["ori_shape"], pbatch["ratio_pad"])
            pbatch["bboxes"] = ops.clip_boxes(orig_boxes, pbatch["ori_shape"])
        return pbatch


# ---------------------------------------------------------------------------
# Candidate Verification & Prepare Gate
# ---------------------------------------------------------------------------
@torch.no_grad()
def prepare(parent, arm: str):
    candidate = build(parent, arm)
    old = parent.state_dict()
    new = candidate.state_dict()
    assert all(k in new and torch.equal(v, new[k]) for k, v in old.items()), (
        "HistoricalCLS parent weight inheritance broken!"
    )

    p2a_ref = candidates_v5.build(parent, "p2_cm0")
    p2a_sd = p2a_ref.state_dict()
    assert all(k in new and torch.equal(v, new[k]) for k, v in p2a_sd.items()), (
        "CLS-P2A parent weight inheritance broken!"
    )

    n = sum(p.numel() for p in candidate.parameters())
    expected_n = 3195674 + ADDED_OVER_HIST[arm]
    assert n == expected_n, f"Param mismatch: got {n}, expected {expected_n}"

    checks: dict[str, Any] = {}
    a, b = copy.deepcopy(p2a_ref).eval(), copy.deepcopy(candidate).eval()
    for shape_tag, (hh, ww) in (("rect_128x160", (128, 160)), ("rect_416x640", (416, 640))):
        x = torch.rand(1, 3, hh, ww, generator=torch.Generator().manual_seed(43))
        ra, rb = a(x)[1], b(x)[1]
        c_sh = {
            branch: {k: float((ra[branch][k] - rb[branch][k]).abs().max()) for k in ("boxes", "scores")}
            for branch in ("one2many", "one2one")
        }
        assert all(val == 0.0 for m in c_sh.values() for val in m.values()), (shape_tag, c_sh)
        checks[f"step0_vs_p2a_{shape_tag}"] = c_sh

    from rebuilt_model import as_o2m

    if has_b1(arm):
        # Check BN params in shallow side
        l2 = candidate.model[2]
        bn_params = [(m.eps, m.momentum) for m in l2.shallow_side.modules() if isinstance(m, nn.BatchNorm2d)]
        assert bn_params == [(0.001, 0.03)] * 2, f"Shallow side BN mismatch: {bn_params}"

        # Explicit 5-branch reference check on DEConv2dBNAct
        c_act = copy.deepcopy(candidate).eval()
        gen = torch.Generator().manual_seed(99)
        l2_act = c_act.model[2]
        l2_act.shallow_side.proj.weight.normal_(0, 0.1, generator=gen)
        l2_act.shallow_side.proj.bias.normal_(0, 0.1, generator=gen)

        deconv = l2_act.shallow_side.deconv
        xd = torch.rand(2, 32, 32, 40, generator=torch.Generator().manual_seed(17))
        y_eq = deconv(xd)
        y_exp = deconv.forward_explicit_branches(xd)
        deconv_branch_err = float((y_eq - y_exp).abs().max())
        assert deconv_branch_err < 1e-5, f"DEConv branch equiv error: {deconv_branch_err}"
        checks["deconv_explicit_branch_err"] = deconv_branch_err

        # Active delta propagates through L2 -> both O2M boxes and scores (including P2..P5)
        x = torch.rand(1, 3, 128, 160, generator=torch.Generator().manual_seed(43))
        r_par = a(x)[1]["one2many"]
        r_act = c_act(x)[1]["one2many"]
        score_delta = float((r_act["scores"] - r_par["scores"]).abs().max())
        box_delta = float((r_act["boxes"] - r_par["boxes"]).abs().max())
        assert score_delta > 1e-3 and box_delta > 1e-3, (score_delta, box_delta)
        checks["active_score_delta_max"] = score_delta
        checks["active_box_delta_max"] = box_delta

        # Unfused vs fused O2M active parity check + DEConv fuse deletion check
        u_o2m = as_o2m(copy.deepcopy(c_act))
        f_o2m = compact_o2m(copy.deepcopy(c_act))
        assert not any(isinstance(m, nn.BatchNorm2d) for m in f_o2m.model[2].shallow_side.modules()), (
            "b1 shallow_side still has unfused BatchNorm2d!"
        )
        assert sum(1 for m in f_o2m.modules() if isinstance(m, nn.BatchNorm2d)) == 20, (
            "Unexpected total residual BN count in fused b1!"
        )
        deconv_fused = f_o2m.model[2].shallow_side.deconv
        for removed in ("vc", "cdc", "adc", "hdc", "vdc", "bn"):
            assert not hasattr(deconv_fused, removed), f"Unremoved DEConv attr after fuse: {removed}"
        assert hasattr(deconv_fused, "conv")
        yu, yf = u_o2m(x)[0], f_o2m(x)[0]
        fuse_box_err = float((yu[:, :4] - yf[:, :4]).abs().max())
        fuse_score_err = float((yu[:, 4:] - yf[:, 4:]).abs().max())
        assert fuse_box_err < 5e-3 and fuse_score_err < 1e-4, (fuse_box_err, fuse_score_err)
        checks["active_fuse_box_err_px"] = fuse_box_err
        checks["active_fuse_score_err"] = fuse_score_err
        checks["deployed_o2m_params"] = sum(p.numel() for p in f_o2m.parameters())
        assert checks["deployed_o2m_params"] == 3054370 + 13440, checks["deployed_o2m_params"]

        # Gradient flow checks: Step-1 proj grad, Step-2 all 5 DEConv branches + reduce grad
        with torch.enable_grad():
            xg = torch.rand(2, 3, 128, 128, generator=torch.Generator().manual_seed(7))
            g1 = copy.deepcopy(candidate).train()
            out1 = g1(xg)
            loss1 = (out1["one2many"]["scores"] ** 2).mean() + (out1["one2many"]["boxes"] ** 2).mean()
            loss1.backward()
            pw = float(g1.model[2].shallow_side.proj.weight.grad.abs().max())
            pb = float(g1.model[2].shallow_side.proj.bias.grad.abs().max())
            assert pw > 0.0 and pb > 0.0, (pw, pb)
            checks["step1_proj_grad"] = dict(weight=pw, bias=pb)

            g2 = copy.deepcopy(candidate).train()
            g2.model[2].shallow_side.proj.weight.data.normal_(0, 0.05, generator=torch.Generator().manual_seed(11))
            out2 = g2(xg)["one2many"]
            (out2["scores"].sum() + out2["boxes"].sum()).backward()
            side2 = g2.model[2].shallow_side
            b_grads = {
                "reduce": float(side2.reduce.conv.weight.grad.abs().max()),
                "vc": float(side2.deconv.vc.weight.grad.abs().max()),
                "cdc": float(side2.deconv.cdc.weight.grad.abs().max()),
                "adc": float(side2.deconv.adc.weight.grad.abs().max()),
                "hdc": float(side2.deconv.hdc.weight.grad.abs().max()),
                "vdc": float(side2.deconv.vdc.weight.grad.abs().max()),
                "proj": float(side2.proj.weight.grad.abs().max()),
            }
            assert min(b_grads.values()) > 0.0, b_grads
            checks["step2_side_grads"] = b_grads
    else:
        f_o2m = compact_o2m(copy.deepcopy(candidate))
        checks["deployed_o2m_params"] = sum(p.numel() for p in f_o2m.parameters())

    if has_warp50(arm):
        # Verify forward/inverse warp round-trip including boundary y=0.5 straddling boxes
        ys = np.array([0.0, 0.25, 0.499, 0.5, 0.501, 0.75, 1.0, -0.05, 1.05], dtype=np.float64)
        rt_np = np.max(np.abs(warp_y_inverse_np(warp_y_forward_np(ys)) - ys))
        rt_t = float(
            (warp_y_inverse_torch(torch.tensor(warp_y_forward_np(ys))) - torch.tensor(ys)).abs().max()
        )
        assert max(rt_np, rt_t) < 1e-12, (rt_np, rt_t)
        checks["warp_roundtrip_max_err"] = float(max(rt_np, rt_t))

    return candidate, dict(
        arm=arm,
        seed=SEEDS[arm],
        params=n,
        added_over_p2a=ADDED_OVER_P2A[arm],
        added=ADDED_OVER_HIST[arm],
        inherited_p2a_tensors=len(p2a_sd),
        protocol_overrides=PROTOCOL_OVERRIDES[arm],
        checks=checks,
    )
