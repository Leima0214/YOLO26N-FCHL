"""YOLO26n/Cls geometry-semantic candidates; SET is training-only.

Task adaptation inspired by TopoSAM's geometry/semantic division, not an
author TSCB/CBF reproduction. Curve sampling reuses the recorded local source.
"""
import copy
import math
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

import rebuilt_model as original
from curve_sampling import RoadSnakeAdapter
from set_training import SETEnhancer, batch_stats_without_running_update
from ultralytics.cfg import get_cfg
from ultralytics.nn.modules import Conv, Detect


PRIMARY = ("Cls_Geom_SET", "Cls")
CONTROLS = ("Native", "Native_Geom", "Cls_Geom", "Native_SET", "Cls_SET", "Native_Geom_SET")
ARMS = PRIMARY + CONTROLS


class GeometryBlock(RoadSnakeAdapter):
    """Local/horizontal/vertical feature extractor, with full 64-channel width."""

    def __init__(self, channels=64):
        super().__init__(channels, kernel_size=7, expansion=1.0,
                         max_offset=1.0, gamma_init=1.0)
        del self.gamma

    def _sample_curve(self, feature, orthogonal_offset, horizontal):
        # Float32 coordinates and cumulative offsets avoid half-grid quantization.
        with torch.autocast(device_type=feature.device.type, enabled=False):
            sampled = super()._sample_curve(feature.float(), orthogonal_offset.float(), horizontal)
        return sampled.to(feature.dtype)

    def forward(self, x):
        reduced = self.reduce(x)
        horizontal, vertical = self.offset(reduced).tanh().chunk(2, 1)
        horizontal = self.act(self.horizontal_norm(self._sample_curve(reduced, horizontal, True)))
        vertical = self.act(self.vertical_norm(self._sample_curve(reduced, vertical, False)))
        return self.fuse(torch.cat((self.local(reduced), horizontal, vertical), 1))


class GeometryPyramid(nn.Module):
    def __init__(self, native):
        super().__init__()
        self.g2 = GeometryBlock(64)
        self.down3 = Conv(64, 64, 3, 2)
        self.p3_projection = Conv(64, 64, 1)
        self.g3 = GeometryBlock(64)
        self.down4 = Conv(64, 128, 3, 2) if native else None

    def forward(self, c2, p3):
        g2 = self.g2(c2)
        g3 = self.g3(self.down3(g2) + self.p3_projection(p3))
        return g2, g3, self.down4(g3) if self.down4 is not None else None


class SemanticResidual(nn.Module):
    """A spatial/channel gate conditions geometry use on local and coarse features."""

    def __init__(self, channels, context_channels):
        super().__init__()
        self.context = Conv(context_channels, 32, 1)
        self.geometry = Conv(channels, channels, 1)
        self.gate = nn.Sequential(nn.Conv2d(2 * channels + 32, 32, 1), nn.SiLU(),
                                  nn.Conv2d(32, channels, 1))
        nn.init.normal_(self.gate[-1].weight, std=0.01)
        nn.init.constant_(self.gate[-1].bias, math.log(0.1 / 0.9))
        self.scale = 1.0  # Runtime intervention for the exact mother-model bypass check.

    def forward(self, local, geometry, context):
        if self.scale == 0:
            return local
        semantic = F.interpolate(self.context(context), size=local.shape[-2:],
                                 mode="bilinear", align_corners=False)
        delta = self.geometry(geometry)
        weight = self.gate(torch.cat((local, delta, semantic), 1)).sigmoid()
        return local + self.scale * weight * delta


class TaskFusion(nn.Module):
    def __init__(self, channels, context_channels):
        super().__init__()
        self.classification = SemanticResidual(channels, context_channels)
        self.localization = SemanticResidual(channels, context_channels)

    def forward(self, local, geometry, cls_context, box_context):
        return (self.classification(local, geometry, cls_context),
                self.localization(local, geometry, box_context))


class FusedDetect(Detect):
    """Shared fusion happens before native O2O detachment; prediction towers stay native."""

    def forward(self, raw):
        c2, p3, p4, p5 = raw
        if getattr(self, "_capture_set", False):
            self._set_features = tuple(raw)
        if self.family == "cls":
            p2 = self.p2_fuse(torch.cat((c2, F.interpolate(p3, size=c2.shape[-2:], mode="nearest")), 1))
            q4, q5 = self.paper2_pki2(p4, p5)
            cls2 = box2 = p2
            cls3 = box3 = p3
            if self.geometry is not None:
                g2, g3, _ = self.geometry(c2, p3)
                cls2, box2 = self.fusions[0](p2, g2, q4, p4)
                cls3, box3 = self.fusions[1](p3, g3, q4, p4)
            streams = [box2, box3, p4, p5, cls2, cls3, q4, q5]
        else:
            cls3 = box3 = p3
            cls4 = box4 = p4
            if self.geometry is not None:
                _, g3, g4 = self.geometry(c2, p3)
                cls3, box3 = self.fusions[0](p3, g3, p4, p4)
                cls4, box4 = self.fusions[1](p4, g4, p5, p5)
            streams = [box3, box4, p5, cls3, cls4, p5]
        return Detect.forward(self, streams)

    def forward_head(self, x, box_head=None, cls_head=None):
        if box_head is None or cls_head is None:
            return {}
        assert len(x) == 2 * self.nl and self.reg_max == 1
        batch = x[0].shape[0]
        boxes = torch.cat([box_head[i](x[i]).view(batch, 4, -1) for i in range(self.nl)], -1)
        scores = torch.cat([cls_head[i](x[i + self.nl]).view(batch, self.nc, -1)
                            for i in range(self.nl)], -1)
        return dict(boxes=boxes, scores=scores, feats=x[:self.nl])


class FusionDetector(original.RebuiltDetector):
    def loss(self, batch, preds=None):
        if not self.training or not self.use_set:
            return super().loss(batch, preds)
        if preds is not None or not torch.is_grad_enabled():
            raise RuntimeError("SET requires integrated forward/loss with feature gradients")
        if getattr(self, "criterion", None) is None:
            self.criterion = self.init_criterion()
        head = self.model[-1]
        head._capture_set = True
        try:
            clean = self.predict(batch["img"])
            features = head._set_features
        finally:
            head._capture_set = False
            head._set_features = None
        clean_loss, clean_items = self.criterion(clean, batch)
        enhanced = self.set_enhancer(features, clean_loss, batch)
        # Recompute geometry, PKI, semantic gates and towers, using the same weights.
        with batch_stats_without_running_update(head):
            auxiliary = head(enhanced)
        auxiliary_loss, auxiliary_items = self.criterion(auxiliary, batch)
        gain = self.set_enhancer.gain
        return clean_loss + gain * auxiliary_loss, clean_items + gain * auxiliary_items


def adapt(model, family, geometry):
    model = copy.deepcopy(model)
    model.__class__ = FusionDetector
    model.use_set = False
    model.criterion = None
    head = model.model[-1]
    head.__class__ = FusedDetect
    head.family = family
    head.f = [2, 16, 19, 22]
    head.geometry = None
    head.fusions = nn.ModuleList()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        if geometry:
            head.geometry = GeometryPyramid(native=family == "native")
            widths = (64, 128) if family == "native" else (64, 64)
            contexts = (128, 256) if family == "native" else (128, 128)
            head.fusions = nn.ModuleList(TaskFusion(c, s) for c, s in zip(widths, contexts))
    head.type = f"GeometrySemantic_{family}"
    model.save = sorted(set(model.save + [2]))
    model.names = dict(enumerate(original.NAMES))
    model.args = get_cfg(overrides=dict(epochs=100, seed=42, imgsz=640,
                                       optimizer="MuSGD", amp=True, compile=False))
    return model


def with_set(model):
    result = copy.deepcopy(model)
    result.use_set = True
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        result.set_enhancer = SETEnhancer(channels=(64, 64, 128, 256), strides=(4, 8, 16, 32))
    return result


def strip_set(model):
    """Keep geometry and fusion at deployment; remove only training-only SET state."""
    model = copy.deepcopy(model).eval()
    model.use_set = False
    if hasattr(model, "set_enhancer"):
        del model.set_enhancer
    model.criterion = None
    return model


def build_arms(checkpoint):
    original.PUBLIC = Path(checkpoint)
    native, cls, unused = original.build_models()
    del unused
    arms = dict(Native=adapt(native, "native", False), Cls=adapt(cls, "cls", False),
                Native_Geom=adapt(native, "native", True), Cls_Geom=adapt(cls, "cls", True))
    for name in tuple(arms):
        arms[name + "_SET"] = with_set(arms[name])
    for base, candidate in (("Native", "Native_Geom"), ("Cls", "Cls_Geom")):
        reference = arms[base].state_dict()
        assert all(torch.equal(value, arms[candidate].state_dict()[key]) for key, value in reference.items())
    for name, model in list(arms.items()):
        if name.endswith("_SET"):
            reference = arms[name[:-4]].state_dict()
            assert all(torch.equal(value, model.state_dict()[key]) for key, value in reference.items())
    return arms
