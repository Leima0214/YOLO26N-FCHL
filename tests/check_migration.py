"""CPU engineering checks with random weights and synthetic tensors, no dataset access."""

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from pycocotools.coco import COCO

from scripts.eval_paper2 import ap, evaluate
from scripts.paper2_common import O2MValidator, file_sha256, install_fixed_schedule, make_o2m_model
from ultralytics.cfg import get_cfg
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.loss import E2ELoss


def main():
    torch.set_num_threads(2)
    torch.manual_seed(42)
    baseline = DetectionModel(str(ROOT / "ultralytics/cfg/models/26/yolo26.yaml"), nc=4, verbose=False)
    strip = DetectionModel(str(ROOT / "ultralytics/cfg/models/26/yolo26n-japan4-s1-strip-regression.yaml"), nc=4, verbose=False)
    strip.load(baseline, verbose=False)
    strip.args = get_cfg()
    sample = torch.randn(1, 3, 64, 64)
    baseline.eval()
    strip.eval()
    with torch.no_grad():
        b0, s1 = baseline(sample), strip(sample)
    for branch in ("one2many", "one2one"):
        for field in ("boxes", "scores"):
            torch.testing.assert_close(b0[1][branch][field], s1[1][branch][field], atol=0, rtol=0)
    print("PASS: B0-to-Strip weight transfer and zero-initialized identity, both heads")

    install_fixed_schedule()
    criterion = strip.init_criterion()
    for _ in range(120):
        criterion.update()
    assert (criterion.o2m, criterion.o2o) == (0.8, 0.2)
    batch = {"batch_idx": torch.zeros(1), "cls": torch.zeros(1, 1),
             "bboxes": torch.tensor([[0.5, 0.5, 0.3, 0.2]])}
    losses, items = criterion(s1, batch)
    assert torch.isfinite(losses).all() and torch.isfinite(items).all()
    assert isinstance(criterion, E2ELoss) and strip.end2end
    print("PASS: fixed 0.8/0.2 schedule and dual-head loss remain available")

    # Make the heads differ so a wrong branch cannot pass through identity initialization.
    with torch.no_grad():
        strip.model[-1].strip_cv2[0].gamma.fill_(0.2)
        for layer in strip.model[-1].one2one_cv3:
            layer[-1].bias.fill_(8)
        native = strip(sample)
        deployed = make_o2m_model(strip)
        o2m = deployed(sample)
    assert strip.end2end and not deployed.end2end
    for field in ("boxes", "scores"):
        torch.testing.assert_close(native[1]["one2many"][field], o2m[1][field])
    assert not torch.equal(native[1]["one2one"]["scores"], native[1]["one2many"]["scores"])
    validator = O2MValidator(args={"conf": 0.00001, "iou": 0.7, "max_det": 300, "plots": False},
                             save_dir=ROOT / "reports/engineering_check")
    validator.training, validator.end2end, validator.box_head = True, False, strip.model[-1]
    train_predictions = validator.postprocess(native)
    validator.training = False
    eval_predictions = validator.postprocess((o2m[0].clone(), o2m[1]))
    for left, right in zip(train_predictions, eval_predictions):
        for field in ("bboxes", "conf", "cls"):
            torch.testing.assert_close(left[field], right[field])
    assert strip.end2end and strip.model[-1].cv2 is not None
    backend = AutoBackend(model=make_o2m_model(strip), device=torch.device("cpu"), fuse=True, verbose=False)
    assert not backend.end2end and backend.model.model[-1].strip_cv2 is not None
    with torch.no_grad():
        fused = backend(sample)
    torch.testing.assert_close(fused[0], o2m[0], atol=1e-4, rtol=1e-4)
    try:
        make_o2m_model(strip.fuse(verbose=False))
    except ValueError:
        pass
    else:
        raise AssertionError("O2O-fused model must be rejected")
    print("PASS: training/standalone O2M selection, NMS and fusion; O2O-only checkpoint rejected")

    gt = COCO()
    gt.dataset = {"images": [{"id": 1, "width": 64, "height": 64, "file_name": "synthetic.jpg"}],
                  "categories": [{"id": 1, "name": "D00"}],
                  "annotations": [{"id": 1, "image_id": 1, "category_id": 1,
                                   "bbox": [10, 10, 10, 10], "area": 100, "iscrowd": 0}]}
    gt.createIndex()
    assert ap(evaluate(gt, [], [1])) == 0.0
    print("PASS: zero detections produce valid COCO metrics")

    manifest = json.loads((ROOT / "docs/migration_manifest.json").read_text(encoding="utf-8"))
    for item in manifest["files"]:
        assert file_sha256(ROOT / item["destination"]) == item["destination_sha256"], item["destination"]
    print(f"PASS: {len(manifest['files'])} migrated files match provenance hashes")


if __name__ == "__main__":
    main()
