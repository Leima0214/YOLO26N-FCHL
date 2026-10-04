"""Explicit Paper 2 protocol; the migrated framework remains unchanged."""

from copy import copy, deepcopy
import hashlib
from pathlib import Path

import torch

from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer, DetectionValidator
from ultralytics.nn.modules.head import Detect
from ultralytics.utils.loss import E2ELoss
from ultralytics.utils.ops import xyxy2xywh


def file_sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def fixed_e2e_update(self):
    self.updates += 1
    self.o2m, self.o2o = 0.8, 0.2


def install_fixed_schedule():
    E2ELoss.update = fixed_e2e_update


def make_o2m_model(model):
    """Copy before AutoBackend fuses away O2M; never modify the training model."""
    head = model.model[-1]
    if not isinstance(head, Detect) or head.cv2 is None or head.cv3 is None:
        raise ValueError("Expected an unfused Detect checkpoint retaining the O2M heads")
    deployed = deepcopy(model)
    for name in ("one2one_cv2", "one2one_cv3", "one2one_strip_cv2"):
        if hasattr(deployed.model[-1], name):
            delattr(deployed.model[-1], name)
    return deployed.eval()


class O2MValidator(DetectionValidator):
    def __call__(self, trainer=None, model=None):
        if trainer is None:
            if model is None or isinstance(model, (str, Path)):
                model = YOLO(str(model or self.args.model), task="detect").model
            model = make_o2m_model(model)
        return super().__call__(trainer=trainer, model=model)

    def init_metrics(self, model):
        super().init_metrics(model)
        self.end2end = False
        if self.training:
            self.box_head = model.model[-1]

    def postprocess(self, preds):
        if self.training:
            # Loss still receives the original dual-head output. Decode only for metrics.
            decoded = self.box_head._inference(preds[1]["one2many"])
            boxes = xyxy2xywh(decoded[:, :4].transpose(1, 2)).transpose(1, 2)
            preds = torch.cat((boxes, decoded[:, 4:]), dim=1)
        return super().postprocess(preds)


class Paper2Trainer(DetectionTrainer):
    def get_model(self, cfg=None, weights=None, verbose=True):
        install_fixed_schedule()  # Also applies inside a spawned DDP worker.
        return super().get_model(cfg=cfg, weights=weights, verbose=verbose)

    def get_validator(self):
        self.loss_names = "box_loss", "cls_loss", "dfl_loss"
        return O2MValidator(self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks)
