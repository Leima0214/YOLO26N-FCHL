"""Step 2A: Frozen-Detector Border-Aware Decoupled Quality Head (BA-QHead, Seed 42).

Trains a ~90K-param BatchNorm-free GroupNorm/LayerNorm visual+boundary quality head
strictly on SVRDD7 Train (6,000 imgs) over a frozen O2M detector:
  - Frozen backbone + neck + cv2/cv3 mid-features ([feat_l, cv2_mid_l, cv3_mid_l])
  - Crack-friendly canonical level routing + dual RoIAlign (inner box vs outer-inner rim)
  - Head 1 (q_iou_head): strictly cls_score-free localization quality regressor & ranker
  - Head 2 (q_cal_head): zero-init residual confidence calibrator with detached visual trunk

Enforces two startup sanity gates before Epoch 1:
  - Gate #1: Train coordinate conversion & GT parity check (mean best same-cls IoU > 0.65)
  - Gate #2: Val native NMS parity <= 0.005 pp AND exact Step-0 q_cal_only equivalence (<= 0.005 pp)

Dual weights saved per Rule #2: best.pt (by Val q_iou_vote_t08 AP50:95), best_ap50.pt (by AP50),
and last.pt (final E15), followed by full decoupled mode, slice, 7-class, latency & transfer evals.
"""
import argparse
import copy
import datetime as dt
import json
import math
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from pycocotools.coco import COCO

STAGE = Path("/root/YOLO26N-FCHL/reports/paper2_cls_candidate_diagnosis_640_20261120_42106")
ROOT = Path("/root/YOLO26N-FCHL")
GEO = ROOT / "reports/paper2_geometry_semantic_640_20261109_42266"
REF = ROOT / "reports/paper2_cls_p2p3_snake_640_20261115_42264"
sys.path[:0] = [
    str(STAGE),
    str(GEO / "source"),
    str(REF / "source/dependencies"),
    str(GEO / "source/dependencies"),
    str(ROOT),
]

import models  # noqa: F401
import experiment as reference
import rebuilt_model
import run_v2
import candidates_v3  # noqa: F401
import candidates_v5  # noqa: F401
import step1_cpu_health_audit as cpu_audit
import step1_box_fusion_replay_s42 as replay
from ultralytics.cfg import get_cfg
from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.data.utils import check_det_dataset
from ultralytics.utils import ops

NAMES = replay.NAMES
ANN_VAL = replay.ANN_VAL
PYRAMID_IN_CHS = (144, 144, 208, 336)
STRIDES = (4.0, 8.0, 16.0, 32.0)


def resolve_stage_path(raw: str) -> Path:
    p = Path(raw)
    return p if p.is_absolute() else STAGE / p


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def box_iou_t(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if a.numel() == 0 or b.numel() == 0:
        return torch.zeros((a.shape[0], b.shape[0]), dtype=torch.float32, device=a.device)
    return torchvision.ops.box_iou(a.float(), b.float())


def rank_corr_stats(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    if len(y_true) < 5 or float(np.std(y_true)) < 1e-9 or float(np.std(y_pred)) < 1e-9:
        return dict(pearson=0.0, spearman=0.0)
    pr = float(np.corrcoef(y_true, y_pred)[0, 1])
    rx = np.argsort(np.argsort(y_true)).astype(np.float64)
    ry = np.argsort(np.argsort(y_pred)).astype(np.float64)
    sr = float(np.corrcoef(rx, ry)[0, 1])
    return dict(pearson=round(pr, 4), spearman=round(sr, 4))


class FrozenDetectorExtractor(nn.Module):
    """Frozen O2M detector exposing pyramid mid-features + decoded boxes/scores/logits."""

    def __init__(self, ckpt_path: Path, device: torch.device):
        super().__init__()
        raw_model = reference.load_model(ckpt_path)
        deployed = rebuilt_model.as_o2m(raw_model).float().eval()
        head = deployed.model[-1]
        if hasattr(head, "ctx_o2o"):
            del head.ctx_o2o
        deployed.fuse(verbose=False)
        deployed = deployed.float().eval().to(device)
        for p in deployed.parameters():
            p.requires_grad_(False)
        head = deployed.model[-1]
        assert len(head.cv2) == 4 and len(head.cv3) == 4, "Expected 4-scale detection head"

        self.deployed = deployed
        self.device = device
        self._cv2 = [None] * 4
        self._cv3 = [None] * 4
        self._hooks = []

        for i in range(4):
            assert len(head.cv2[i]) == 3 and len(head.cv3[i]) == 3, f"Unexpected head tower depth at scale {i}"
            assert head.cv2[i][1].conv.out_channels == 16
            assert head.cv3[i][1][1].conv.out_channels == 64

            def _make_hook(buf, idx):
                def _hook(_mod, _inp, out):
                    buf[idx] = out.detach()
                return _hook

            self._hooks.append(head.cv2[i][1].register_forward_hook(_make_hook(self._cv2, i)))
            self._hooks.append(head.cv3[i][1].register_forward_hook(_make_hook(self._cv3, i)))

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    @torch.no_grad()
    def forward(self, imgs: torch.Tensor) -> dict:
        for i in range(4):
            self._cv2[i] = None
            self._cv3[i] = None
        decoded, meta = self.deployed(imgs)
        feats_4 = [f.detach().float() for f in meta["feats"]]
        assert len(feats_4) == 4
        pyramid_in_4 = []
        for i in range(4):
            assert self._cv2[i] is not None and self._cv3[i] is not None, f"Hook missed level {i}"
            cat = torch.cat([feats_4[i], self._cv2[i].detach().float(), self._cv3[i].detach().float()], dim=1)
            assert cat.shape[1] == PYRAMID_IN_CHS[i], f"Ch mismatch at {i}: {cat.shape[1]}"
            pyramid_in_4.append(cat)
        for i in range(4):
            self._cv2[i] = None
            self._cv3[i] = None

        levels = torch.cat(
            [torch.full((int(v.shape[-2] * v.shape[-1]),), k, dtype=torch.int8, device=imgs.device) for k, v in enumerate(feats_4, 2)]
        )
        boxes_640 = ops.xywh2xyxy(decoded[:, :4].detach().float().transpose(1, 2))
        scores_7 = decoded[:, 4:].detach().float().transpose(1, 2)
        logits_7 = meta["scores"].detach().float().transpose(1, 2)
        return dict(
            pyramid_in_4=pyramid_in_4,
            boxes_640=boxes_640,
            scores_7=scores_7,
            logits_7=logits_7,
            levels=levels,
        )


class BorderAwareQualityHead(nn.Module):
    """Lightweight GN/LN visual+boundary quality regressor and decoupled confidence calibrator."""

    def __init__(self, nc: int = 7):
        super().__init__()
        self.nc = nc
        self.level_projs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(in_ch, 32, 1, bias=False),
                    nn.GroupNorm(8, 32),
                    nn.SiLU(inplace=True),
                    nn.Conv2d(32, 32, 3, padding=1, groups=32, bias=False),
                    nn.GroupNorm(8, 32),
                    nn.SiLU(inplace=True),
                    nn.Conv2d(32, 32, 1, bias=False),
                    nn.GroupNorm(8, 32),
                    nn.SiLU(inplace=True),
                )
                for in_ch in PYRAMID_IN_CHS
            ]
        )
        self.vis_tower = nn.Sequential(
            nn.Conv2d(64, 48, 3, padding=0, bias=False),
            nn.GroupNorm(8, 48),
            nn.SiLU(inplace=True),
            nn.Conv2d(48, 48, 3, padding=0, bias=False),
            nn.GroupNorm(8, 48),
            nn.SiLU(inplace=True),
        )
        # Geometry (8) + routed level onehot (4) + class onehot (7) = 19
        self.geom_enc = nn.Sequential(
            nn.Linear(19, 32),
            nn.LayerNorm(32),
            nn.SiLU(inplace=True),
        )
        # Head 1: strictly cls_score-free localization IoU head
        self.q_iou_head = nn.Sequential(
            nn.Linear(80, 48),
            nn.LayerNorm(48),
            nn.SiLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(48, 1),
        )
        # Head 2: zero-init residual calibration head with detached visual/IoU trunk
        self.q_cal_head = nn.Sequential(
            nn.Linear(85, 48),
            nn.LayerNorm(48),
            nn.SiLU(inplace=True),
            nn.Dropout(0.15),
            nn.Linear(48, 1),
        )
        nn.init.zeros_(self.q_cal_head[-1].weight)
        nn.init.zeros_(self.q_cal_head[-1].bias)

        assert not any(isinstance(m, torch.nn.modules.batchnorm._BatchNorm) for m in self.modules()), (
            "BA-QHead must not contain any BatchNorm layers!"
        )

    def encode_pyramid(self, pyramid_in_4: list[torch.Tensor]) -> list[torch.Tensor]:
        return [proj(feat) for proj, feat in zip(self.level_projs, pyramid_in_4)]

    @staticmethod
    def _route_levels(boxes_xyxy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        w = (boxes_xyxy[:, 2] - boxes_xyxy[:, 0]).clamp_min(1.0)
        h = (boxes_xyxy[:, 3] - boxes_xyxy[:, 1]).clamp_min(1.0)
        eff_scale = torch.minimum(torch.sqrt(w * h), 2.5 * torch.minimum(w, h))
        lvl = torch.zeros_like(eff_scale, dtype=torch.long)
        lvl = torch.where(eff_scale >= 48.0, torch.ones_like(lvl), lvl)
        lvl = torch.where(eff_scale >= 96.0, torch.full_like(lvl, 2), lvl)
        lvl = torch.where(eff_scale >= 192.0, torch.full_like(lvl, 3), lvl)
        return lvl, w, h

    def forward_candidates(
        self,
        q_feats: list[torch.Tensor],
        img_hw: tuple[int, int],
        batch_idx: torch.Tensor,
        boxes_xyxy: torch.Tensor,
        cls_idx: torch.Tensor,
        cls_logit: torch.Tensor | None = None,
        cl_feats: torch.Tensor | None = None,
        compute_cal: bool = True,
        logit_noise_std: float = 0.0,
    ) -> dict[str, torch.Tensor]:
        n = boxes_xyxy.shape[0]
        device = boxes_xyxy.device
        if n == 0:
            z = torch.zeros((0,), dtype=torch.float32, device=device)
            return dict(z_iou=z, q_iou=z, delta_logit=z, s_cal=z)

        img_h, img_w = float(img_hw[0]), float(img_hw[1])
        x1 = boxes_xyxy[:, 0].clamp(0.0, img_w - 1.0)
        y1 = boxes_xyxy[:, 1].clamp(0.0, img_h - 1.0)
        x2 = torch.maximum(boxes_xyxy[:, 2].clamp(0.0, img_w), x1 + 1.0)
        y2 = torch.maximum(boxes_xyxy[:, 3].clamp(0.0, img_h), y1 + 1.0)
        b_clamped = torch.stack([x1, y1, x2, y2], dim=1)

        routed_lvl, w, h = self._route_levels(b_clamped)
        pad_x = torch.clamp(0.15 * w, min=2.0)
        pad_y = torch.clamp(0.15 * h, min=2.0)
        ox1 = (x1 - pad_x).clamp(0.0, img_w - 1.0)
        oy1 = (y1 - pad_y).clamp(0.0, img_h - 1.0)
        ox2 = torch.maximum((x2 + pad_x).clamp(0.0, img_w), ox1 + 1.0)
        oy2 = torch.maximum((y2 + pad_y).clamp(0.0, img_h), oy1 + 1.0)
        b_outer = torch.stack([ox1, oy1, ox2, oy2], dim=1)

        b_idx_f = batch_idx.float().unsqueeze(1)
        rois_in_all = torch.cat([b_idx_f, b_clamped], dim=1)
        rois_out_all = torch.cat([b_idx_f, b_outer], dim=1)

        inner_feat = torch.zeros((n, 32, 5, 5), dtype=torch.float32, device=device)
        outer_feat = torch.zeros((n, 32, 5, 5), dtype=torch.float32, device=device)
        for l_idx, stride in enumerate(STRIDES):
            mask = routed_lvl == l_idx
            if torch.any(mask):
                sc = 1.0 / stride
                inner_feat[mask] = torchvision.ops.roi_align(
                    q_feats[l_idx], rois_in_all[mask], output_size=(5, 5), spatial_scale=sc, sampling_ratio=2, aligned=True
                )
                outer_feat[mask] = torchvision.ops.roi_align(
                    q_feats[l_idx], rois_out_all[mask], output_size=(5, 5), spatial_scale=sc, sampling_ratio=2, aligned=True
                )

        vis_in = torch.cat([inner_feat, outer_feat - inner_feat], dim=1)
        vis_emb = self.vis_tower(vis_in).flatten(1)  # [N, 48]

        cx = 0.5 * (x1 + x2) / img_w
        cy = 0.5 * (y1 + y2) / img_h
        wn = w / img_w
        hn = h / img_h
        log_w = torch.log((w + 1.0) / img_w)
        log_h = torch.log((h + 1.0) / img_h)
        log_ar = torch.log((w + 1.0) / (h + 1.0))
        min_s = torch.minimum(w, h) / img_w
        lvl_oh = F.one_hot(routed_lvl, num_classes=4).float()
        cls_oh = F.one_hot(cls_idx.long(), num_classes=self.nc).float()
        geom_in = torch.cat([torch.stack([cx, cy, wn, hn, log_w, log_h, log_ar, min_s], dim=1), lvl_oh, cls_oh], dim=1)
        g_emb = self.geom_enc(geom_in)  # [N, 32]

        z_iou = self.q_iou_head(torch.cat([vis_emb, g_emb], dim=1)).squeeze(1)
        q_iou = torch.sigmoid(z_iou)

        if not compute_cal:
            return dict(z_iou=z_iou, q_iou=q_iou)

        assert cls_logit is not None and cl_feats is not None
        logit_in = cls_logit
        if self.training and logit_noise_std > 0.0:
            logit_in = cls_logit + torch.randn_like(cls_logit) * logit_noise_std
        cal_in = torch.cat(
            [vis_emb.detach(), g_emb.detach(), q_iou.detach().unsqueeze(1), logit_in.unsqueeze(1), cl_feats],
            dim=1,
        )
        delta_logit = self.q_cal_head(cal_in).squeeze(1)
        s_cal = torch.sigmoid(cls_logit + delta_logit)
        return dict(z_iou=z_iou, q_iou=q_iou, delta_logit=delta_logit, s_cal=s_cal)


@torch.no_grad()
def build_gpu_c70_clusters(
    boxes_xyxy: torch.Tensor, scores: torch.Tensor, cls_idx: torch.Tensor, iou_thres: float = 0.70
) -> dict[str, torch.Tensor]:
    """Vectorized GPU greedy C70 cluster membership matching batched_nms suppression."""
    n = scores.shape[0]
    device = boxes_xyxy.device
    order = torch.argsort(scores, descending=True, stable=True)
    b = boxes_xyxy[order]
    sc = scores[order]
    c = cls_idx[order]

    kept_local = torchvision.ops.batched_nms(b.float(), sc.float(), c.long(), iou_thres)
    k_len = kept_local.shape[0]
    iou_mat = box_iou_t(b[kept_local], b)  # [K, N]
    same_cls = c[kept_local].unsqueeze(1) == c.unsqueeze(0)
    before_or_self = kept_local.unsqueeze(1) <= torch.arange(n, device=device).unsqueeze(0)
    eligible = before_or_self & same_cls & (iou_mat > iou_thres)

    # First kept anchor with iou > thres & k <= j is exact greedy suppressor
    has_hit = torch.any(eligible, dim=0)
    first_k_idx = torch.argmax(eligible.to(torch.int8), dim=0)
    # Fallback self/nearest kept anchor if rare tie occurs
    if not torch.all(has_hit):
        miss_mask = ~has_hit
        fallback_k = torch.argmax((iou_mat * same_cls.float())[:, miss_mask], dim=0)
        first_k_idx[miss_mask] = fallback_k

    cl_size_per_k = torch.bincount(first_k_idx, minlength=k_len).float()
    cl_size = cl_size_per_k[first_k_idx]
    anchor_local = kept_local[first_k_idx]
    anchor_score = sc[anchor_local].clamp_min(1e-6)
    iou_to_anchor = iou_mat[first_k_idx, torch.arange(n, device=device)]

    # Un-permute back or keep sorted order cleanly
    return dict(
        order=order,
        boxes=b,
        scores=sc,
        cls_idx=c,
        cluster_id=first_k_idx,
        cl_size=cl_size,
        anchor_score=anchor_score,
        iou_to_anchor=iou_to_anchor,
    )


def sample_jitter_boxes(
    gt_boxes: torch.Tensor, gt_cls: torch.Tensor, img_hw: tuple[int, int]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate 12 synthetic jittered boxes per GT (6 fine + 6 medium/coarse) with exact same-cls max IoU."""
    g = gt_boxes.shape[0]
    device = gt_boxes.device
    if g == 0:
        e4 = torch.zeros((0, 4), dtype=torch.float32, device=device)
        e1 = torch.zeros((0,), dtype=torch.float32, device=device)
        el = torch.zeros((0,), dtype=torch.long, device=device)
        return e4, el, e1, el

    img_h, img_w = float(img_hw[0]), float(img_hw[1])
    bw = (gt_boxes[:, 2] - gt_boxes[:, 0]).clamp_min(1.0)
    bh = (gt_boxes[:, 3] - gt_boxes[:, 1]).clamp_min(1.0)

    def _jit(k: int, frac: float, px: float) -> torch.Tensor:
        rep = gt_boxes.unsqueeze(1).expand(g, k, 4).clone()
        rw = bw.view(g, 1, 1)
        rh = bh.view(g, 1, 1)
        dx = (torch.rand((g, k, 2), device=device) * 2.0 - 1.0) * (frac * rw + px)
        dy = (torch.rand((g, k, 2), device=device) * 2.0 - 1.0) * (frac * rh + px)
        rep[:, :, 0] += dx[:, :, 0]
        rep[:, :, 2] += dx[:, :, 1]
        rep[:, :, 1] += dy[:, :, 0]
        rep[:, :, 3] += dy[:, :, 1]
        return rep

    fine = _jit(6, 0.18, 2.5)
    coarse = _jit(6, 0.45, 3.0)
    all_j = torch.cat([fine, coarse], dim=1).reshape(-1, 4)

    x1 = all_j[:, 0].clamp(0.0, img_w - 1.0)
    y1 = all_j[:, 1].clamp(0.0, img_h - 1.0)
    x2 = torch.maximum(all_j[:, 2].clamp(0.0, img_w), x1 + 1.0)
    y2 = torch.maximum(all_j[:, 3].clamp(0.0, img_h), y1 + 1.0)
    j_boxes = torch.stack([x1, y1, x2, y2], dim=1)
    j_cls = gt_cls.unsqueeze(1).expand(g, 12).reshape(-1)
    j_grp = torch.arange(g, device=device).unsqueeze(1).expand(g, 12).reshape(-1)

    iou_mat = box_iou_t(j_boxes, gt_boxes)  # [12G, G]
    same_c = j_cls.unsqueeze(1) == gt_cls.unsqueeze(0)
    y_iou = (iou_mat * same_c.float()).max(dim=1).values
    return j_boxes, j_cls, y_iou, j_grp


def compute_rank_loss(q_list: list[torch.Tensor], y_list: list[torch.Tensor]) -> torch.Tensor:
    """Vectorized padded pairwise ranking loss across heterogeneous groups."""
    valid_q = []
    valid_y = []
    for q, y in zip(q_list, y_list):
        if len(q) >= 2 and float(y.max().item()) >= 0.35:
            valid_q.append(q[:32])
            valid_y.append(y[:32])
    if not valid_q:
        return torch.zeros((), dtype=torch.float32, device=q_list[0].device if q_list else "cuda")

    device = valid_q[0].device
    max_k = max(len(x) for x in valid_q)
    ng = len(valid_q)
    qp = torch.zeros((ng, max_k), dtype=torch.float32, device=device)
    yp = torch.zeros((ng, max_k), dtype=torch.float32, device=device)
    vm = torch.zeros((ng, max_k), dtype=torch.bool, device=device)
    for i, (q, y) in enumerate(zip(valid_q, valid_y)):
        k = len(q)
        qp[i, :k] = q
        yp[i, :k] = y
        vm[i, :k] = True

    dy = yp.unsqueeze(2) - yp.unsqueeze(1)
    dq = qp.unsqueeze(2) - qp.unsqueeze(1)
    pair_mask = vm.unsqueeze(2) & vm.unsqueeze(1) & (dy >= 0.03)
    if not torch.any(pair_mask):
        return torch.zeros((), dtype=torch.float32, device=device)
    w = dy[pair_mask]
    losses = w * F.softplus(-6.0 * dq[pair_mask])
    return losses.sum() / w.sum().clamp_min(1e-6)


def build_train_loader(batch_size: int = 32, workers: int = 4, aug: bool = True, shuffle: bool = True):
    data_dict = check_det_dataset(str(run_v2.DATA_YAML))
    overrides = dict(
        task="detect",
        imgsz=640,
        batch=batch_size,
        rect=False,
        workers=workers,
        data=str(run_v2.DATA_YAML),
        mosaic=0.0,
        mixup=0.0,
        copy_paste=0.0,
        degrees=0.0,
        perspective=0.0,
        fliplr=0.5 if aug else 0.0,
        flipud=0.0,
        translate=0.08 if aug else 0.0,
        scale=0.15 if aug else 0.0,
        hsv_h=0.015 if aug else 0.0,
        hsv_s=0.4 if aug else 0.0,
        hsv_v=0.3 if aug else 0.0,
    )
    cfg = get_cfg(overrides=overrides)
    ds = build_yolo_dataset(cfg, data_dict["train"], batch=batch_size, data=data_dict, mode="train" if aug else "val", stride=32)
    assert len(ds) == 6000, f"Expected 6000 Train images, got {len(ds)}"
    return build_dataloader(ds, batch=batch_size, workers=workers, shuffle=shuffle, rank=-1, drop_last=False, pin_memory=True)


def extract_train_batch_gt(batch: dict, img_hw: tuple[int, int], device: torch.device) -> list[tuple[torch.Tensor, torch.Tensor]]:
    img_h, img_w = float(img_hw[0]), float(img_hw[1])
    bsz = int(batch["img"].shape[0])
    b_idx = batch["batch_idx"].to(device).long()
    cls_all = batch["cls"].to(device).long().view(-1)
    boxes_n = batch["bboxes"].to(device).float()
    if len(boxes_n):
        scale_t = torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32, device=device)
        boxes_px = ops.xywh2xyxy(boxes_n * scale_t)
        boxes_px[:, [0, 2]] = boxes_px[:, [0, 2]].clamp(0.0, img_w)
        boxes_px[:, [1, 3]] = boxes_px[:, [1, 3]].clamp(0.0, img_h)
    else:
        boxes_px = torch.zeros((0, 4), dtype=torch.float32, device=device)

    out = []
    for b in range(bsz):
        m = b_idx == b
        out.append((boxes_px[m], cls_all[m]))
    return out


@torch.no_grad()
def sanity_gate_1_train_coords(extractor: FrozenDetectorExtractor, device: torch.device) -> float:
    loader = build_train_loader(batch_size=32, workers=0, aug=False, shuffle=False)
    batch = next(iter(loader))
    imgs = batch["img"].to(device).float() / 255.0
    det = extractor(imgs)
    gts = extract_train_batch_gt(batch, imgs.shape[-2:], device)
    ious = []
    for b, (gt_b, gt_c) in enumerate(gts):
        if len(gt_b) == 0:
            continue
        b640 = det["boxes_640"][b]
        sc = det["scores_7"][b]
        keep = sc.max(dim=1).values > 0.001
        if not torch.any(keep):
            ious.extend([0.0] * len(gt_b))
            continue
        bk = b640[keep]
        sk = sc[keep]
        imat = box_iou_t(gt_b, bk)
        best = imat.max(dim=1).values
        ious.extend(best.cpu().tolist())
    mean_iou = float(np.mean(ious))
    print(f"[Gate #1 Train Coord Parity] GTs={len(ious)}, mean_best_IoU={mean_iou:.4f}", flush=True)
    assert mean_iou > 0.65, f"Train coord gate failed: {mean_iou:.4f} <= 0.65"
    return mean_iou


def prepare_val_cache(ckpt_path: Path, extractor: FrozenDetectorExtractor, official_metrics_json: Path | None) -> dict:
    """Capture exact TaskValidator Val batches + precomputed C70 clusters & parity check."""
    model_val = reference.load_model(ckpt_path)
    tmp_dir = STAGE / "diagnostics/_tmp_qhead_val_cache"
    v = replay.RawCaptureValidator(
        args=dict(
            task="detect",
            data=str(run_v2.DATA_YAML),
            imgsz=640,
            batch=32,
            workers=4,
            device="0",
            conf=0.001,
            iou=0.7,
            max_det=300,
            rect=True,
            save_json=True,
            plots=False,
            verbose=False,
            half=False,
            split="val",
        ),
        save_dir=tmp_dir,
    )
    v.captured_images = {}
    gt = COCO(str(ANN_VAL))
    v.ground_truth = gt
    v.image_ids = {Path(im["file_name"]).stem: k for k, im in gt.imgs.items()}
    v.categories = {c["name"]: k for k, c in gt.cats.items()}
    v(model=model_val)
    assert len(v.captured_images) == 1000

    cats_by_name = v.categories
    cat_ids_list = [cats_by_name[n] for n in NAMES]
    official_preds = []
    for p in v.jdict:
        idx = int(p["category_id"]) - 1
        official_preds.append(
            dict(
                image_id=v.image_ids[Path(p["file_name"]).stem],
                category_id=cats_by_name[reference.NAMES[idx]],
                bbox=p["bbox"],
                score=p["score"],
            )
        )
    ev_off = cpu_audit.official_eval(gt, cpu_audit.build_coco_dt(gt, official_preds), cat_ids_list)
    off_m = {k: round(v * 100, 4) for k, v in cpu_audit.extract_coco_metrics(ev_off, 0).items()}

    if official_metrics_json is not None and official_metrics_json.is_file():
        ref_m = json.loads(official_metrics_json.read_text())["main"]
        for k in ("AP50_95", "AP50", "AP75"):
            d = abs(off_m[k] - ref_m[k] * 100.0)
            assert d <= 0.005, f"Official metrics parity diff on {k}: {off_m[k]} vs {ref_m[k]*100.0}"

    # Build GT maps in 1024 coords
    cname_to_idx = {n: i for i, n in enumerate(NAMES)}
    cat_id_to_idx = {cats_by_name[n]: cname_to_idx[n] for n in NAMES}
    gt_boxes_by_img = defaultdict(list)
    gt_np = defaultdict(lambda: defaultdict(list))
    for ann in gt.anns.values():
        if ann.get("iscrowd") or ann.get("ignore"):
            continue
        x, y, w, h = ann["bbox"]
        box = np.array([x, y, x + w, y + h], dtype=np.float32)
        cidx = cat_id_to_idx[ann["category_id"]]
        gt_boxes_by_img[ann["image_id"]].append((cidx, box))
        gt_np[ann["image_id"]][cidx].append(box)
    gt_np_arr = {im: {c: np.stack(arr, axis=0) for c, arr in cd.items()} for im, cd in gt_np.items()}

    # Iterate exact validator dataloader to cache Val batches + raw extractor candidates
    cached_batches = []
    raw_by_img = {}
    for batch in v.dataloader:
        imgs_cpu = batch["img"].clone()
        batch = v.preprocess(batch)
        imgs = batch["img"]
        det = extractor(imgs)
        b_items = []
        for si, name in enumerate(batch["im_file"]):
            pb = v._prepare_batch(si, batch)
            image_id = v.image_ids[Path(name).stem]
            b640_t = det["boxes_640"][si]
            sc_t = det["scores_7"][si]
            lg_t = det["logits_7"][si]
            keep_anchor = sc_t.max(dim=1).values > 0.001
            k_indices = torch.nonzero(keep_anchor, as_tuple=True)[0]

            b1024 = ops.scale_boxes(pb["imgsz"], b640_t[k_indices].clone(), pb["ori_shape"], pb["ratio_pad"]).cpu().numpy().astype(np.float32)
            b640 = b640_t[k_indices].cpu().numpy().astype(np.float32)
            sc = sc_t[k_indices].cpu().numpy().astype(np.float32)
            lg = lg_t[k_indices].cpu().numpy().astype(np.float32)
            lv = det["levels"][k_indices].cpu().numpy().astype(np.int8)

            # Build exact (a_idx, c_idx) candidate list (>0.001, sorted descending, <=30000) + greedy C70 suppressors
            a_idx, c_idx = np.nonzero(sc > 0.001)
            cand_sc = sc[a_idx, c_idx]
            order = np.argsort(-cand_sc, kind="mergesort")[:30000]
            a_idx = a_idx[order]
            c_idx = c_idx[order]
            cand_sc = cand_sc[order]
            cand_lg = lg[a_idx, c_idx]
            cand_b640 = b640[a_idx]
            cand_b1024 = b1024[a_idx]
            cand_lv = lv[a_idx]

            clusters, cand_cl_feats = build_val_clusters_and_feats(
                cand_b640, cand_b1024, cand_sc, cand_lv, c_idx, gt_np_arr.get(image_id, {})
            )

            raw_by_img[image_id] = dict(
                image_id=int(image_id),
                fname=Path(name).name,
                boxes_640=b640,
                boxes_1024=b1024,
                scores=sc,
                levels=lv,
                cand_b640=cand_b640,
                cand_b1024=cand_b1024,
                cand_sc=cand_sc,
                cand_lg=cand_lg,
                cand_cls=c_idx,
                cand_cl_feats=cand_cl_feats,
                clusters=clusters,
            )
            b_items.append(
                dict(
                    si=si,
                    image_id=int(image_id),
                    cand_b640=torch.from_numpy(cand_b640),
                    cand_cls=torch.from_numpy(c_idx).long(),
                    cand_lg=torch.from_numpy(cand_lg),
                    cand_cl_feats=torch.from_numpy(cand_cl_feats),
                )
            )
        cached_batches.append(dict(imgs_cpu=imgs_cpu, img_hw=imgs.shape[-2:], items=b_items))

    return dict(
        gt=gt,
        cats_by_name=cats_by_name,
        cat_ids_list=cat_ids_list,
        gt_boxes_by_img=gt_boxes_by_img,
        gt_np=gt_np_arr,
        off_m=off_m,
        cached_batches=cached_batches,
        raw_by_img=raw_by_img,
    )


def build_val_clusters_and_feats(
    cand_b640: np.ndarray,
    cand_b1024: np.ndarray,
    cand_sc: np.ndarray,
    cand_lv: np.ndarray,
    cand_cls: np.ndarray,
    gt_cls_dict: dict[int, np.ndarray],
    iou_thres: float = 0.70,
    max_det: int = 300,
) -> tuple[list[dict], np.ndarray]:
    """Exact greedy C70 clusters storing global candidate indices + per-candidate cluster features."""
    n = len(cand_sc)
    cl_feats = np.zeros((n, 3), dtype=np.float32)
    if n == 0:
        return [], cl_feats

    by_class = defaultdict(list)
    for pos in range(n):
        by_class[int(cand_cls[pos])].append(pos)

    kept_clusters_by_pos = {}
    for c, pos_list in by_class.items():
        pos_arr = np.array(pos_list, dtype=np.int64)
        c_b640 = cand_b640[pos_arr]
        c_b1024 = cand_b1024[pos_arr]
        c_sc = cand_sc[pos_arr]
        c_lv = cand_lv[pos_arr]
        iou_mat = replay.iou_matrix_xyxy(c_b640, c_b640)

        # Ground-truth IoU for diagnostic oracle / ranking metrics
        gts = gt_cls_dict.get(c)
        if gts is not None and len(gts):
            gt_ious = replay.iou_matrix_xyxy(c_b1024, gts).max(axis=1)
        else:
            gt_ious = np.zeros((len(pos_arr),), dtype=np.float32)

        suppressed_by = [-1] * len(pos_arr)
        survivor_local = []
        for i in range(len(pos_arr)):
            if suppressed_by[i] != -1:
                continue
            survivor_local.append(i)
            suppressed_by[i] = i
            hits = np.flatnonzero(iou_mat[i, i + 1 :] > iou_thres) + (i + 1)
            for j in hits:
                if suppressed_by[j] == -1:
                    suppressed_by[j] = i

        sup_arr = np.array(suppressed_by, dtype=np.int64)
        for i in survivor_local:
            c70_local = np.flatnonzero(sup_arr == i)
            cl_sz = len(c70_local)
            a_sc = max(float(c_sc[i]), 1e-6)
            log_sz = math.log1p(cl_sz) / 3.0
            for m in c70_local:
                gp = int(pos_arr[m])
                cl_feats[gp, 0] = log_sz
                cl_feats[gp, 1] = float(c_sc[m]) / a_sc
                cl_feats[gp, 2] = float(iou_mat[i, m])

            g_pos = int(pos_arr[i])
            kept_clusters_by_pos[g_pos] = dict(
                class_idx=c,
                anchor_cand_idx=g_pos,
                anchor_score=float(c_sc[i]),
                anchor_level=int(c_lv[i]),
                anchor_box_1024=c_b1024[i].copy(),
                c70_cand_idxs=pos_arr[c70_local],
                c70_boxes_1024=c_b1024[c70_local],
                c70_scores=c_sc[c70_local],
                c70_gt_ious=gt_ious[c70_local],
            )

    clusters = [kept_clusters_by_pos[g] for g in sorted(kept_clusters_by_pos.keys())[:max_det]]
    return clusters, cl_feats


@torch.no_grad()
def predict_val_candidates(
    val_cache: dict, extractor: FrozenDetectorExtractor, qhead: BorderAwareQualityHead, device: torch.device
) -> dict[int, dict]:
    qhead.eval()
    extractor.eval()
    preds_by_img = {}
    for batch in val_cache["cached_batches"]:
        imgs = batch["imgs_cpu"].to(device).float() / 255.0
        det = extractor(imgs)
        q_feats = qhead.encode_pyramid(det["pyramid_in_4"])
        items = batch["items"]
        non_empty = [it for it in items if len(it["cand_b640"]) > 0]
        if non_empty:
            b_idx_cat = torch.cat([torch.full((len(it["cand_b640"]),), it["si"], dtype=torch.long, device=device) for it in non_empty])
            box_cat = torch.cat([it["cand_b640"] for it in non_empty]).to(device)
            cls_cat = torch.cat([it["cand_cls"] for it in non_empty]).to(device)
            lg_cat = torch.cat([it["cand_lg"] for it in non_empty]).to(device)
            clf_cat = torch.cat([it["cand_cl_feats"] for it in non_empty]).to(device)
            out = qhead.forward_candidates(q_feats, batch["img_hw"], b_idx_cat, box_cat, cls_cat, lg_cat, clf_cat, compute_cal=True)
            q_all = out["q_iou"].cpu().numpy().astype(np.float32)
            s_all = out["s_cal"].cpu().numpy().astype(np.float32)
            offset = 0
            for it in items:
                k = len(it["cand_b640"])
                if k == 0:
                    preds_by_img[it["image_id"]] = dict(q_iou=np.zeros((0,), np.float32), s_cal=np.zeros((0,), np.float32))
                else:
                    preds_by_img[it["image_id"]] = dict(q_iou=q_all[offset : offset + k], s_cal=s_all[offset : offset + k])
                    offset += k
        else:
            for it in items:
                preds_by_img[it["image_id"]] = dict(q_iou=np.zeros((0,), np.float32), s_cal=np.zeros((0,), np.float32))
    return preds_by_img


def compute_quality_diagnostics(val_cache: dict, qpreds_by_img: dict[int, dict]) -> dict:
    ys, qs = [], []
    pair_correct = 0
    pair_total = 0
    top1_iou_gains = []
    for img_id, rec in val_cache["raw_by_img"].items():
        q_arr = qpreds_by_img[img_id]["q_iou"]
        for cl in rec["clusters"]:
            gt_i = cl["c70_gt_ious"]
            if len(gt_i) == 0:
                continue
            qc = q_arr[cl["c70_cand_idxs"]]
            ys.append(gt_i)
            qs.append(qc)
            if float(gt_i.max()) >= 0.35 and len(gt_i) >= 2:
                pick = int(np.argmax(qc))
                top1_iou_gains.append(float(gt_i[pick] - gt_i[0]))
                dy = gt_i[:, None] - gt_i[None, :]
                dq = qc[:, None] - qc[None, :]
                pm = dy >= 0.05
                if np.any(pm):
                    pair_total += int(pm.sum())
                    pair_correct += int((dq[pm] > 0.0).sum())

    y_cat = np.concatenate(ys) if ys else np.zeros((0,), np.float32)
    q_cat = np.concatenate(qs) if qs else np.zeros((0,), np.float32)
    m30 = y_cat >= 0.30
    m50 = y_cat >= 0.50
    corr = rank_corr_stats(y_cat[m30], q_cat[m30])
    mae50 = float(np.mean(np.abs(y_cat[m50] - q_cat[m50]))) if np.any(m50) else 0.0
    return dict(
        c70_pair_acc_dy05=round(pair_correct / max(pair_total, 1), 4),
        c70_pair_total=pair_total,
        mean_top1_cluster_iou_gain_ge35=round(float(np.mean(top1_iou_gains)) if top1_iou_gains else 0.0, 5),
        mae_iou_ge50=round(mae50, 4),
        **{f"{k}_ge30": v for k, v in corr.items()},
    )


def decode_mode_preds(
    val_cache: dict, qpreds_by_img: dict[int, dict], mode: str
) -> tuple[list[dict], dict[int, list[dict]], dict[int, list[dict]]]:
    cat_ids_list = val_cache["cat_ids_list"]
    coco_preds = []
    by_img_xyxy = {}
    eff_clusters = {}

    for img_id, rec in val_cache["raw_by_img"].items():
        q_arr = qpreds_by_img[img_id]["q_iou"]
        s_cal_arr = qpreds_by_img[img_id]["s_cal"]
        resolved = []
        img_clusters = []

        if mode in ("native_nms", "q_iou_select", "q_iou_vote_t08", "q_iou_vote_cls", "q_cal_only", "q_cal_plus_vote"):
            clusters_for_img = []
            for cl in rec["clusters"]:
                idxs = cl["c70_cand_idxs"]
                b70 = cl["c70_boxes_1024"]
                sc70 = cl["c70_scores"]
                qc = q_arr[idxs]
                sc_cal = s_cal_arr[idxs]
                a_sc = cl["anchor_score"]

                if mode == "native_nms":
                    box = cl["anchor_box_1024"]
                    out_sc = a_sc
                elif mode == "q_iou_select":
                    best_k = int(np.argmax(qc)) if len(qc) else 0
                    box = b70[best_k]
                    out_sc = a_sc
                elif mode == "q_iou_vote_t08":
                    if len(b70) == 1:
                        box = b70[0]
                    else:
                        z = (qc - qc.max()) / 0.08
                        w = np.exp(z)
                        w = (w / np.maximum(w.sum(), 1e-12))[:, None]
                        box = (b70 * w).sum(axis=0)
                    out_sc = a_sc
                elif mode == "q_iou_vote_cls":
                    if len(b70) == 1:
                        box = b70[0]
                    else:
                        w = ((qc ** 2) * sc70)[:, None]
                        wsum = w.sum()
                        box = (b70 * w).sum(axis=0) / wsum if wsum > 1e-12 else b70[0]
                    out_sc = a_sc
                elif mode == "q_cal_only":
                    box = cl["anchor_box_1024"]
                    out_sc = float(sc_cal[0])
                elif mode == "q_cal_plus_vote":
                    if len(b70) == 1:
                        box = b70[0]
                    else:
                        z = (qc - qc.max()) / 0.08
                        w = np.exp(z)
                        w = (w / np.maximum(w.sum(), 1e-12))[:, None]
                        box = (b70 * w).sum(axis=0)
                    out_sc = float(sc_cal[0])
                else:
                    raise ValueError(mode)

                clusters_for_img.append((out_sc, cl, box))

            # If scores changed (q_cal_*), re-sort descending by out_sc
            if mode.startswith("q_cal"):
                clusters_for_img.sort(key=lambda t: -t[0])

            for out_sc, cl, box in clusters_for_img[:300]:
                x1, y1, x2, y2 = [float(v) for v in box]
                resolved.append(
                    dict(
                        class_idx=cl["class_idx"],
                        score=float(out_sc),
                        bbox_xyxy=box.astype(np.float32),
                        bbox_coco=[x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)],
                    )
                )
                img_clusters.append(cl)
        elif mode in ("q_nms_order_keep_score", "q_score_blend_b025"):
            cand_b640 = rec["cand_b640"]
            cand_b1024 = rec["cand_b1024"]
            cand_sc = rec["cand_sc"]
            cand_cls = rec["cand_cls"]
            if len(cand_sc):
                if mode == "q_nms_order_keep_score":
                    rank_sc = (np.maximum(cand_sc, 1e-8) ** 0.65) * (np.maximum(q_arr, 1e-6) ** 0.35)
                else:
                    rank_sc = (np.maximum(cand_sc, 1e-8) ** 0.75) * (np.maximum(q_arr, 1e-6) ** 0.25)
                order = np.argsort(-rank_sc, kind="mergesort")
                boxes_off = torch.from_numpy(cand_b640[order] + cand_cls[order].astype(np.float32)[:, None] * 7680.0)
                kept_local = torchvision.ops.nms(boxes_off, torch.from_numpy(rank_sc[order]), 0.70).numpy()[:300]
                kept = order[kept_local]
                out_scores = cand_sc[kept] if mode == "q_nms_order_keep_score" else rank_sc[kept]
                # Sort by final output score descending for COCO top-100 consistency
                fin_ord = np.argsort(-out_scores, kind="mergesort")
                for idx_k in kept[fin_ord]:
                    box = cand_b1024[idx_k]
                    out_sc = float(cand_sc[idx_k] if mode == "q_nms_order_keep_score" else rank_sc[idx_k])
                    x1, y1, x2, y2 = [float(v) for v in box]
                    resolved.append(
                        dict(
                            class_idx=int(cand_cls[idx_k]),
                            score=out_sc,
                            bbox_xyxy=box.astype(np.float32),
                            bbox_coco=[x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)],
                        )
                    )
                img_clusters = rec["clusters"]
        else:
            raise ValueError(mode)

        by_img_xyxy[img_id] = resolved
        eff_clusters[img_id] = img_clusters
        for r in resolved:
            coco_preds.append(
                dict(
                    image_id=int(img_id),
                    category_id=cat_ids_list[r["class_idx"]],
                    bbox=[round(float(v), 3) for v in r["bbox_coco"]],
                    score=round(float(r["score"]), 5),
                )
            )
    return coco_preds, by_img_xyxy, eff_clusters


def evaluate_val_modes(
    val_cache: dict,
    qpreds_by_img: dict[int, dict],
    modes: tuple[str, ...],
    fast_main_only: bool = False,
) -> dict:
    gt = val_cache["gt"]
    cats_by_name = val_cache["cats_by_name"]
    cat_ids_list = val_cache["cat_ids_list"]
    gt_boxes_by_img = val_cache["gt_boxes_by_img"]
    results = {}

    for mname in modes:
        coco_preds, by_img_xyxy, eff_clusters = decode_mode_preds(val_cache, qpreds_by_img, mname)
        if fast_main_only:
            ev = cpu_audit.official_eval(gt, cpu_audit.build_coco_dt(gt, coco_preds), cat_ids_list)
            m0 = cpu_audit.extract_coco_metrics(ev, 0)
            results[mname] = dict(
                main={k: round(float(m0[k] * 100.0), 3) for k in ("AP50", "AP75", "AP50_95", "AR100")}
            )
        else:
            res = replay.evaluate_prediction_set(gt, cats_by_name, coco_preds, by_img_xyxy, gt_boxes_by_img)
            if mname in ("native_nms", "q_iou_vote_t08", "q_nms_order_keep_score"):
                cl_map = {img_id: rec["clusters"] for img_id, rec in val_cache["raw_by_img"].items()}
                res["unmatched75_breakdown"] = replay.analyze_unmatched75_breakdown(
                    gt, cats_by_name, val_cache["raw_by_img"], cl_map, coco_preds
                )
            results[mname] = res
    return results


@torch.no_grad()
def benchmark_latency_fp32(
    extractor: FrozenDetectorExtractor, qhead: BorderAwareQualityHead, device: torch.device, batch_size: int = 1
) -> dict:
    """Measure pure FP32 GPU inference speed (bs=1 and bs=32) for baseline vs + BA-QHead."""
    extractor.eval()
    qhead.eval()
    for bs_tag, bs, n_reps in (("bs1", 1, 150), ("bs32", 32, 40)):
        x = torch.rand(bs, 3, 640, 640, device=device)
        # Warmup
        for _ in range(10):
            det = extractor(x)
            qf = qhead.encode_pyramid(det["pyramid_in_4"])
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_reps):
            _ = extractor.deployed(x)
        torch.cuda.synchronize()
        base_ms = (time.perf_counter() - t0) * 1000.0 / (n_reps * bs)

        # Realistic ~250 candidates per image across batch
        k_per_img = 250
        b_idx = torch.arange(bs, device=device).repeat_interleave(k_per_img)
        fake_boxes = torch.tensor([[100.0, 200.0, 140.0, 220.0]], device=device).repeat(bs * k_per_img, 1)
        fake_cls = torch.zeros((bs * k_per_img,), dtype=torch.long, device=device)
        fake_lg = torch.zeros((bs * k_per_img,), dtype=torch.float32, device=device)
        fake_clf = torch.zeros((bs * k_per_img, 3), dtype=torch.float32, device=device)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_reps):
            det = extractor(x)
            qf = qhead.encode_pyramid(det["pyramid_in_4"])
            _ = qhead.forward_candidates(qf, (640, 640), b_idx, fake_boxes, fake_cls, fake_lg, fake_clf, compute_cal=True)
        torch.cuda.synchronize()
        full_ms = (time.perf_counter() - t0) * 1000.0 / (n_reps * bs)
        if bs_tag == "bs1":
            out_bs1 = dict(base_ms_img=round(base_ms, 3), full_ms_img=round(full_ms, 3), added_ms_img=round(full_ms - base_ms, 3))
        else:
            out_bs32 = dict(base_ms_img=round(base_ms, 3), full_ms_img=round(full_ms, 3), added_ms_img=round(full_ms - base_ms, 3))
    return dict(fp32_bs1=out_bs1, fp32_bs32=out_bs32)


def train_one_epoch(
    extractor: FrozenDetectorExtractor,
    qhead: BorderAwareQualityHead,
    train_loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict:
    qhead.train()
    extractor.eval()

    acc = defaultdict(float)
    n_batches = 0
    for batch in train_loader:
        imgs = batch["img"].to(device, non_blocking=True).float() / 255.0
        img_hw = imgs.shape[-2:]
        bsz = imgs.shape[0]
        with torch.no_grad():
            det = extractor(imgs)
            gts = extract_train_batch_gt(batch, img_hw, device)

        # Gather per-image real candidates (score > 0.002, top-300) + C70 clusters + synthetic jitter
        real_bidx, real_boxes, real_cls, real_lg, real_clf, real_yiou, real_tcal = [], [], [], [], [], [], []
        jit_bidx, jit_boxes, jit_cls, jit_yiou = [], [], [], []
        real_rank_slices = []  # (global_indices, y_iou) will resolve after concat
        jit_rank_slices = []

        real_total = 0
        jit_total = 0
        with torch.no_grad():
            for b in range(bsz):
                gt_b, gt_c = gts[b]
                sc7 = det["scores_7"][b]
                lg7 = det["logits_7"][b]
                b640 = det["boxes_640"][b]

                a_idx, c_idx = torch.nonzero(sc7 > 0.002, as_tuple=True)
                if len(a_idx):
                    csc = sc7[a_idx, c_idx]
                    clg = lg7[a_idx, c_idx]
                    if len(csc) > 300:
                        _, top_i = torch.topk(csc, 300, sorted=True)
                        a_idx, c_idx, csc, clg = a_idx[top_i], c_idx[top_i], csc[top_i], clg[top_i]
                    cl_res = build_gpu_c70_clusters(b640[a_idx], csc, c_idx, iou_thres=0.70)
                    ord_i = cl_res["order"]
                    rb = cl_res["boxes"]
                    rc = cl_res["cls_idx"]
                    rsc = cl_res["scores"]
                    rlg = clg[ord_i]
                    rclf = torch.stack(
                        [
                            torch.log1p(cl_res["cl_size"]) / 3.0,
                            rsc / cl_res["anchor_score"],
                            cl_res["iou_to_anchor"],
                        ],
                        dim=1,
                    )
                    if len(gt_b):
                        imat = box_iou_t(rb, gt_b)
                        same_c = rc.unsqueeze(1) == gt_c.unsqueeze(0)
                        yiou = (imat * same_c.float()).max(dim=1).values
                    else:
                        yiou = torch.zeros((len(rb),), dtype=torch.float32, device=device)
                    tcal = torch.where(yiou >= 0.25, yiou, torch.zeros_like(yiou))

                    # Collect C70 clusters of size >= 2 and max y_iou >= 0.35
                    cl_id = cl_res["cluster_id"]
                    uniq_ids, counts = torch.unique(cl_id, return_counts=True)
                    for uid, cnt in zip(uniq_ids.tolist(), counts.tolist()):
                        if cnt >= 2:
                            m = cl_id == uid
                            yg = yiou[m]
                            if float(yg.max().item()) >= 0.35:
                                idxs = torch.nonzero(m, as_tuple=True)[0] + real_total
                                real_rank_slices.append((idxs, yg))

                    n_r = len(rb)
                    real_bidx.append(torch.full((n_r,), b, dtype=torch.long, device=device))
                    real_boxes.append(rb)
                    real_cls.append(rc)
                    real_lg.append(rlg)
                    real_clf.append(rclf)
                    real_yiou.append(yiou)
                    real_tcal.append(tcal)
                    real_total += n_r

                jb, jc, jy, jgrp = sample_jitter_boxes(gt_b, gt_c, img_hw)
                if len(jb):
                    n_j = len(jb)
                    g_cnt = len(gt_b)
                    jit_bidx.append(torch.full((n_j,), b, dtype=torch.long, device=device))
                    jit_boxes.append(jb)
                    jit_cls.append(jc)
                    jit_yiou.append(jy)
                    for gi in range(g_cnt):
                        s = jit_total + gi * 12
                        idxs = torch.arange(s, s + 12, dtype=torch.long, device=device)
                        jit_rank_slices.append((idxs, jy[gi * 12 : (gi + 1) * 12]))
                    jit_total += n_j

        optimizer.zero_grad(set_to_none=True)
        q_feats = qhead.encode_pyramid(det["pyramid_in_4"])

        # Real forward (IoU + Cal)
        loss_iou_reg = torch.zeros((), dtype=torch.float32, device=device)
        loss_rank = torch.zeros((), dtype=torch.float32, device=device)
        loss_cal = torch.zeros((), dtype=torch.float32, device=device)

        if real_total > 0:
            r_bi = torch.cat(real_bidx)
            r_bx = torch.cat(real_boxes)
            r_cl = torch.cat(real_cls)
            r_lg = torch.cat(real_lg)
            r_cf = torch.cat(real_clf)
            r_y = torch.cat(real_yiou)
            r_tc = torch.cat(real_tcal)
            out_r = qhead.forward_candidates(
                q_feats, img_hw, r_bi, r_bx, r_cl, r_lg, r_cf, compute_cal=True, logit_noise_std=0.35
            )
            rq, rz = out_r["q_iou"], out_r["z_iou"]
            rcal_s, rcal_d = out_r["s_cal"], out_r["delta_logit"]

            # L_cal strictly on real candidates
            cal_logits = r_lg + rcal_d
            focal_w = torch.abs(r_tc - rcal_s.detach()).pow(1.5)
            bce_cal = F.binary_cross_entropy_with_logits(cal_logits, r_tc, reduction="none")
            loss_cal = 0.3 * (focal_w * bce_cal).mean() + 1e-3 * (rcal_d ** 2).mean()
        else:
            rq = rz = r_y = torch.zeros((0,), dtype=torch.float32, device=device)

        if jit_total > 0:
            j_bi = torch.cat(jit_bidx)
            j_bx = torch.cat(jit_boxes)
            j_cl = torch.cat(jit_cls)
            j_y = torch.cat(jit_yiou)
            out_j = qhead.forward_candidates(q_feats, img_hw, j_bi, j_bx, j_cl, compute_cal=False)
            jq, jz = out_j["q_iou"], out_j["z_iou"]
        else:
            jq = jz = j_y = torch.zeros((0,), dtype=torch.float32, device=device)

        # Combine positive/semi-pos regression set (real >= 0.15 U jittered)
        pos_mask_r = r_y >= 0.15
        q_pos = torch.cat([rq[pos_mask_r], jq])
        z_pos = torch.cat([rz[pos_mask_r], jz])
        y_pos = torch.cat([r_y[pos_mask_r], j_y])
        if len(y_pos):
            loss_iou_reg = F.smooth_l1_loss(q_pos, y_pos, beta=0.05) + 0.5 * F.binary_cross_entropy_with_logits(z_pos, y_pos)

        # Sampled zero-IoU real negatives (1:1 up to n_pos)
        neg_idx = torch.nonzero(r_y == 0.0, as_tuple=True)[0]
        if len(neg_idx) and len(y_pos):
            if len(neg_idx) > len(y_pos):
                perm = torch.randperm(len(neg_idx), device=device)[: len(y_pos)]
                neg_idx = neg_idx[perm]
            q_neg = rq[neg_idx]
            loss_iou_reg = loss_iou_reg + 0.25 * F.smooth_l1_loss(q_neg, torch.zeros_like(q_neg), beta=0.05)

        # Rank loss within real C70 clusters + synthetic GT jitter groups
        rk_qs = [rq[idxs] for idxs, _ in real_rank_slices] + [jq[idxs] for idxs, _ in jit_rank_slices]
        rk_ys = [yg for _, yg in real_rank_slices] + [yg for _, yg in jit_rank_slices]
        if rk_qs:
            loss_rank = compute_rank_loss(rk_qs, rk_ys)

        loss = loss_iou_reg + 1.5 * loss_rank + loss_cal
        if not loss.requires_grad:
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(qhead.parameters(), 10.0)
        optimizer.step()

        acc["loss"] += float(loss.item())
        acc["loss_iou_reg"] += float(loss_iou_reg.item())
        acc["loss_rank"] += float(loss_rank.item())
        acc["loss_cal"] += float(loss_cal.item())
        n_batches += 1

    return {k: round(v / max(n_batches, 1), 5) for k, v in acc.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", default="qhead_p2a")
    parser.add_argument("--base_ckpt", default="runs/HistCLS_SchemeA_P2/weights/best.pt")
    parser.add_argument("--base_metrics_json", default="records/HistCLS_SchemeA_P2/evaluation/best/metrics.json")
    parser.add_argument("--eval_transfer_ckpt", default="runs/HistCLS_SchemeA_P2/weights/ema_E60_90_bn_recal.pt")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda:0")
    base_ck = resolve_stage_path(args.base_ckpt)
    base_mj = resolve_stage_path(args.base_metrics_json) if args.base_metrics_json else None
    run_dir = STAGE / "runs" / args.run_name
    w_dir = run_dir / "weights"
    w_dir.mkdir(parents=True, exist_ok=True)

    extractor = FrozenDetectorExtractor(base_ck, device=device)
    qhead = BorderAwareQualityHead(nc=len(NAMES)).float().to(device)
    n_added = sum(p.numel() for p in qhead.parameters())
    n_base = sum(p.numel() for p in extractor.deployed.parameters())
    print(f"[{args.run_name}] Base O2M params={n_base:,} | BA-QHead params={n_added:,}", flush=True)

    # Gate #1: Clean Train coordinate sanity check
    train_gate_iou = sanity_gate_1_train_coords(extractor, device)

    # Gate #2: Val native NMS parity + exact Step-0 q_cal equivalence
    val_cache = prepare_val_cache(base_ck, extractor, base_mj)
    q0_preds = predict_val_candidates(val_cache, extractor, qhead, device)
    step0_modes = evaluate_val_modes(val_cache, q0_preds, ("native_nms", "q_cal_only", "q_iou_vote_t08"), fast_main_only=True)
    nat0 = step0_modes["native_nms"]["main"]
    cal0 = step0_modes["q_cal_only"]["main"]
    off0 = val_cache["off_m"]
    d_nat = max(abs(nat0[k] - off0[k]) for k in ("AP50_95", "AP50", "AP75"))
    d_cal0 = max(abs(cal0[k] - nat0[k]) for k in ("AP50_95", "AP50", "AP75"))
    print(f"[Gate #2 Val Parity] native={nat0} vs off={off0} (max_diff={d_nat:.4f} pp) | Step0 q_cal diff={d_cal0:.4f} pp", flush=True)
    assert d_nat <= 0.005, f"Val native NMS parity failed: {d_nat}"
    assert d_cal0 <= 0.005, f"Step-0 q_cal zero-init parity failed: {d_cal0}"

    train_loader = build_train_loader(batch_size=args.batch, workers=4, aug=True, shuffle=True)
    optimizer = torch.optim.AdamW(qhead.parameters(), lr=args.lr, weight_decay=1e-2)

    def lr_lambda(ep: int) -> float:
        if ep == 0:
            return 0.5
        t = (ep - 1) / max(args.epochs - 1, 1)
        return 0.01 + 0.99 * 0.5 * (1.0 + math.cos(math.pi * t))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    best_ap95 = -1.0
    best_ap50 = -1.0
    best_ep = -1
    best_ap50_ep = -1
    epoch_log = []

    for ep in range(1, args.epochs + 1):
        t0 = time.perf_counter()
        tr_stats = train_one_epoch(extractor, qhead, train_loader, optimizer, device)
        scheduler.step()
        train_sec = time.perf_counter() - t0

        t1 = time.perf_counter()
        qpreds = predict_val_candidates(val_cache, extractor, qhead, device)
        qdiag = compute_quality_diagnostics(val_cache, qpreds)
        ev_fast = evaluate_val_modes(val_cache, qpreds, ("q_iou_vote_t08", "q_iou_select", "q_cal_only"), fast_main_only=True)
        eval_sec = time.perf_counter() - t1

        v_vote = ev_fast["q_iou_vote_t08"]["main"]
        v_sel = ev_fast["q_iou_select"]["main"]
        v_cal = ev_fast["q_cal_only"]["main"]

        rec = dict(
            epoch=ep,
            lr=round(optimizer.param_groups[0]["lr"], 6),
            train_sec=round(train_sec, 2),
            eval_sec=round(eval_sec, 2),
            train=tr_stats,
            quality=qdiag,
            val_q_iou_vote_t08=v_vote,
            val_q_iou_select=v_sel,
            val_q_cal_only=v_cal,
        )
        epoch_log.append(rec)

        ckpt_payload = dict(
            qhead=copy.deepcopy(qhead.state_dict()),
            epoch=ep,
            base_ckpt=str(base_ck),
            base_sha256=reference.file_sha(base_ck),
            seed=args.seed,
            qhead_params=n_added,
            metrics=v_vote,
            quality=qdiag,
        )
        if v_vote["AP50_95"] > best_ap95:
            best_ap95 = v_vote["AP50_95"]
            best_ep = ep
            torch.save(ckpt_payload, w_dir / "best.pt")
        if v_vote["AP50"] > best_ap50:
            best_ap50 = v_vote["AP50"]
            best_ap50_ep = ep
            torch.save(ckpt_payload, w_dir / "best_ap50.pt")
        torch.save(ckpt_payload, w_dir / "last.pt")

        print(
            f"[Ep {ep:02d}/{args.epochs:02d}] {train_sec:.1f}s+{eval_sec:.1f}s loss={tr_stats['loss']:.4f} "
            f"(reg={tr_stats['loss_iou_reg']:.4f} rk={tr_stats['loss_rank']:.4f} cal={tr_stats['loss_cal']:.4f}) | "
            f"pairAcc={qdiag['c70_pair_acc_dy05']:.3f} r={qdiag['spearman_ge30']:.3f} gain={qdiag['mean_top1_cluster_iou_gain_ge35']:+.4f} | "
            f"vote={v_vote['AP50_95']:.3f}/{v_vote['AP50']:.3f}/{v_vote['AP75']:.3f} sel={v_sel['AP50_95']:.3f} cal={v_cal['AP50_95']:.3f}/{v_cal['AP50']:.3f}",
            flush=True,
        )

    # Full official-equivalent evaluation of best.pt, best_ap50.pt, last.pt across all modes
    all_modes = (
        "native_nms",
        "q_iou_select",
        "q_iou_vote_t08",
        "q_iou_vote_cls",
        "q_nms_order_keep_score",
        "q_score_blend_b025",
        "q_cal_only",
        "q_cal_plus_vote",
    )

    full_evals = {}
    for wtag in ("best", "best_ap50", "last"):
        wpath = w_dir / f"{wtag}.pt"
        pay = torch.load(wpath, map_location="cpu")
        qhead.load_state_dict(pay["qhead"], strict=True)
        qpreds = predict_val_candidates(val_cache, extractor, qhead, device)
        qdiag = compute_quality_diagnostics(val_cache, qpreds)
        modes_res = evaluate_val_modes(val_cache, qpreds, all_modes, fast_main_only=False)
        full_evals[wtag] = dict(
            checkpoint=str(wpath),
            checkpoint_sha256=reference.file_sha(wpath),
            selected_epoch=int(pay["epoch"]),
            quality=qdiag,
            modes=modes_res,
        )
        print(f"=== Full Eval [{args.run_name}/{wtag} E{pay['epoch']}] qdiag={qdiag} ===", flush=True)
        for m in all_modes:
            mm = modes_res[m]["main"]
            sl = modes_res[m]["slices"]
            print(
                f"  {m:24s}: 95={mm['AP50_95']:.3f} 50={mm['AP50']:.3f} 75={mm['AP75']:.3f} AR={mm['AR100']:.3f} "
                f"| lt8={sl['short640_lt8']['AP50_95']:.3f} 8_16={sl['short640_8_16']['AP50_95']:.3f} lctc={sl['LC_TC_short640_lt16']['AP50_95']:.3f}",
                flush=True,
            )

    latency = benchmark_latency_fp32(extractor, qhead, device)
    print(f"[Latency FP32] {latency}", flush=True)

    # Optional zero-shot transfer onto ema_E60_90_bn_recal.pt
    transfer_eval = {}
    if args.eval_transfer_ckpt:
        t_ck = resolve_stage_path(args.eval_transfer_ckpt)
        if t_ck.is_file():
            extractor.remove_hooks()
            t_ext = FrozenDetectorExtractor(t_ck, device=device)
            t_cache = prepare_val_cache(t_ck, t_ext, None)
            for wtag in ("best", "last"):
                pay = torch.load(w_dir / f"{wtag}.pt", map_location="cpu")
                qhead.load_state_dict(pay["qhead"], strict=True)
                tq = predict_val_candidates(t_cache, t_ext, qhead, device)
                t_diag = compute_quality_diagnostics(t_cache, tq)
                t_modes = evaluate_val_modes(t_cache, tq, all_modes, fast_main_only=False)
                transfer_eval[wtag] = dict(
                    transfer_base_ckpt=str(t_ck),
                    transfer_base_sha256=reference.file_sha(t_ck),
                    qhead_epoch=int(pay["epoch"]),
                    quality=t_diag,
                    modes=t_modes,
                )
                print(f"=== Zero-Shot Transfer [{args.run_name}/{wtag} -> {t_ck.name}] qdiag={t_diag} ===", flush=True)
                for m in all_modes:
                    mm = t_modes[m]["main"]
                    print(f"  {m:24s}: 95={mm['AP50_95']:.3f} 50={mm['AP50']:.3f} 75={mm['AP75']:.3f} AR={mm['AR100']:.3f}", flush=True)
            t_ext.remove_hooks()

    summary = dict(
        generated_at=dt.datetime.now().isoformat(),
        run_name=args.run_name,
        seed=args.seed,
        epochs=args.epochs,
        base_ckpt=str(base_ck),
        base_sha256=reference.file_sha(base_ck),
        base_o2m_params=n_base,
        qhead_params=n_added,
        startup_gates=dict(
            train_coord_mean_best_iou=round(train_gate_iou, 4),
            val_native_parity_diff_pp=round(d_nat, 4),
            val_step0_qcal_diff_pp=round(d_cal0, 4),
        ),
        best_epoch_ap95=best_ep,
        best_epoch_ap50=best_ap50_ep,
        latency_fp32=latency,
        epochs_history=epoch_log,
        eval=full_evals,
        zero_shot_transfer=transfer_eval,
    )

    out_path = STAGE / "diagnostics" / f"{args.run_name}_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[DONE] Saved {out_path}", flush=True)


if __name__ == "__main__":
    main()
