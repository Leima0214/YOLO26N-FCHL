"""Reconstructed P2/PKI2-Cls with task-specific residual P3 ASFF.
Historical tensor identities are not assumed. Author PKSBlock is vendored unchanged.
"""
import copy
import hashlib
import math
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.nn.modules import C3k2, Conv, Detect
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.torch_utils import init_seeds
from pkinet_v2_author import PKSBlock

ROOT = Path('/root/YOLO26N-FCHL')
STAGE = Path(__file__).resolve().parent.parent
NAMES = ['LC', 'TC', 'AC', 'P', 'MC', 'LP', 'TP']
PUBLIC = Path('/root/pretrained/yolo26n.pt')
PUBLIC_SHA = '9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef'


def file_sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def state_sha(module):
    result = hashlib.sha256()
    for name, value in module.state_dict().items():
        result.update(name.encode())
        result.update(value.detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()


class PKIClassification(nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(2))
        self.stages = nn.ModuleList([nn.Sequential(PKSBlock(c), PKSBlock(c)) for c in [128, 256]])

    def forward(self, p4, p5):
        return [x + self.alpha[i] * (self.stages[i](x) - x) for i, x in enumerate([p4, p5])]


class ASFFP3(nn.Module):
    """ASFF mechanism adapted to YOLO26 widths and native Conv/SiLU; P3/P4/P5 order."""
    def __init__(self):
        super().__init__()
        self.compress_p4 = Conv(128, 64, 1)
        self.compress_p5 = Conv(256, 64, 1)
        self.weight_features = nn.ModuleList([Conv(64, 16, 1) for _ in range(3)])
        self.weight_levels = nn.Conv2d(48, 3, 1)
        self.expand = Conv(64, 64, 3)
        self.capture = False
        self.observation = {}

    def forward(self, p3, p4, p5):
        resized = [p3, F.interpolate(self.compress_p4(p4), size=p3.shape[-2:], mode='nearest'),
                   F.interpolate(self.compress_p5(p5), size=p3.shape[-2:], mode='nearest')]
        logits = self.weight_levels(torch.cat([m(x) for m, x in zip(self.weight_features, resized)], 1))
        weights = F.softmax(logits.float(), 1).to(p3.dtype)
        mixed = resized[0] * weights[:, 0:1] + resized[1] * weights[:, 1:2] + resized[2] * weights[:, 2:3]
        out = self.expand(mixed)
        if self.capture:
            w = weights.detach().float()
            self.observation = {'weight_P3_P4_P5': w.mean((0, 2, 3)).cpu().tolist(),
                                'entropy': float(-(w * w.clamp_min(1e-8).log()).sum(1).mean()),
                                'delta_over_raw_rms': float((out-p3).float().square().mean().sqrt() /
                                                            p3.float().square().mean().sqrt().clamp_min(1e-8))}
            self.capture = False
        return out


class P2TaskDetect(Detect):
    def __init__(self):
        super().__init__(nc=7, reg_max=1, end2end=True, ch=(64, 64, 128, 256))
        self.stride = torch.tensor([4., 8., 16., 32.])
        self.p2_fuse = C3k2(128, 64, n=1, c3k=False, e=0.5)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(42)
            self.paper2_pki2 = PKIClassification()
        self.asff = None
        self.task_gate_bias = None

    def add_task_asff(self):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(42)
            self.asff = ASFFP3()
        # Nonzero start: both ASFF and task gates receive gradients on the first step.
        self.task_gate_bias = nn.Parameter(torch.full((2,), math.log(0.1 / 0.9)))

    def forward(self, raw):
        c2, p3, p4, p5 = raw
        p2 = self.p2_fuse(torch.cat([c2, F.interpolate(p3, size=c2.shape[-2:], mode='nearest')], 1))
        cls4, cls5 = self.paper2_pki2(p4, p5)
        cls3 = reg3 = p3
        if self.asff is not None:
            delta = self.asff(p3, p4, p5) - p3
            gates = self.task_gate_bias.sigmoid().to(p3.dtype)
            cls3 = p3 + gates[0] * delta
            reg3 = p3 + gates[1] * delta
        # Native Detect.forward detaches all eight tensors before the O2O towers.
        return super().forward([p2, reg3, p4, p5, p2, cls3, cls4, cls5])

    def forward_head(self, x, box_head=None, cls_head=None):
        if box_head is None or cls_head is None:
            return {}
        assert len(x) == 8 and self.nl == 4
        batch = x[0].shape[0]
        boxes = torch.cat([box_head[i](x[i]).view(batch, 4, -1) for i in range(4)], -1)
        scores = torch.cat([cls_head[i](x[i+4]).view(batch, self.nc, -1) for i in range(4)], -1)
        return {'boxes': boxes, 'scores': scores, 'feats': x[:4]}


class RebuiltDetector(DetectionModel):
    def init_criterion(self):
        loss = super().init_criterion()
        # Preserve historical small-box assignment thresholds [8,16], while anchors use all four true strides.
        for branch in [loss.one2many, loss.one2one]:
            branch.assigner.stride = [8., 16., 32.]
        return loss


def build_models():
    assert file_sha(PUBLIC) == PUBLIC_SHA
    init_seeds(42, deterministic=True)
    public = YOLO(str(PUBLIC)).model.float()
    native = DetectionModel(copy.deepcopy(public.yaml), nc=7, verbose=False).float()
    native.load(public, verbose=False)
    assert sum(p.numel() for p in native.parameters()) == 2506530
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        cls = RebuiltDetector(copy.deepcopy(native.yaml), nc=7, verbose=False).float()
        cls.load_state_dict(native.state_dict(), strict=True)
        old = cls.model[-1]
        head = P2TaskDetect()
        head.bias_init()
        for name in ['cv2', 'cv3', 'one2one_cv2', 'one2one_cv3']:
            for i in range(3):
                getattr(head, name)[i+1].load_state_dict(getattr(old, name)[i].state_dict(), strict=True)
        head.i, head.f, head.type = 23, [2, 16, 19, 22], 'RebuiltP2PKI2Cls'
        cls.model[-1] = head
        cls.save = sorted(set(cls.save + [2]))
        cls.stride = head.stride.clone()
        cls.names = dict(enumerate(NAMES))
        cls.args = get_cfg(overrides={'epochs': 100, 'seed': 42})
    candidate = copy.deepcopy(cls)
    candidate.model[-1].add_task_asff()
    candidate.model[-1].type = 'TaskResidualASFFP3'
    assert sum(p.numel() for p in cls.parameters()) == 3195674
    assert sum(p.numel() for p in candidate.parameters()) == 3260815
    shared = candidate.state_dict()
    assert all(torch.equal(v, shared[k]) for k, v in cls.state_dict().items())
    return native, cls, candidate


def as_o2m(model):
    deployed = copy.deepcopy(model).float().eval()
    head = deployed.model[-1]
    assert head.cv2 is not None
    del head.one2one_cv2, head.one2one_cv3
    assert not head.end2end
    return deployed
