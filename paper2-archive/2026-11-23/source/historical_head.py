"""October classification-only head recovered from the recorded original source."""
import torch
from torch.nn import functional as F
from ultralytics.nn.modules.head import Detect
from models import FusedDetect
from rebuilt_model import PKIClassification


class HistoricalPKI(PKIClassification):
    def forward(self, features):
        p3, p4, p5 = features
        return [p3, *super().forward(p4, p5)]


class P2PKI2ClsDetect(FusedDetect):
    def forward(self, features):
        c2, p3, p4, p5 = features
        # Compute once before the native O2O detach, as in the shared-input control.
        context = self.paper2_pki2([p3, p4, p5])
        p2 = self.p2_fusion(torch.cat((c2, F.interpolate(p3, size=c2.shape[-2:], mode='nearest')), 1))
        boxes = [p2, p3, p4, p5]
        classes = [p2, *context]
        # Detect.forward detaches all eight tensors for O2O; outputs still expose four feats.
        return Detect.forward(self, boxes + classes)

    def forward_head(self, x, box_head=None, cls_head=None):
        if box_head is None or cls_head is None:
            return {}
        assert len(x) == 8 and self.nl == 4
        bs = x[0].shape[0]
        boxes = torch.cat([box_head[i](x[i]).view(bs, 4*self.reg_max, -1) for i in range(4)], -1)
        scores = torch.cat([cls_head[i](x[i+4]).view(bs, self.nc, -1) for i in range(4)], -1)
        return dict(boxes=boxes, scores=scores, feats=x[:4])


def bind(model):
    head = model.model[-1]
    if isinstance(head, P2PKI2ClsDetect):
        return model
    assert head.family == "cls" and head.geometry is None and not model.use_set
    assert "p2_fuse" in head._modules and "p2_fusion" not in head._modules
    head._modules = {("p2_fusion" if k == "p2_fuse" else k): v for k, v in head._modules.items()}
    head.paper2_pki2.__class__ = HistoricalPKI
    head.__class__ = P2PKI2ClsDetect
    return model
