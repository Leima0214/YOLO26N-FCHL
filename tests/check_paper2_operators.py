"""One engineering check for the actual nc=4 C/A/R topology, including CUDA AMP."""

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.ops import xywh2xyxy
from ultralytics.utils.torch_utils import init_seeds
from scripts.diagnose_japan4_head_candidates import branch_gaps, decode_branch, operator_gains
from scripts.paper2_common import O2MValidator, make_o2m_model, o2m_state_sha256
from scripts.train_paper2 import MODELS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    device = torch.device(args.device)
    init_seeds(42, deterministic=True)
    # Compare fusion in FP32 arithmetic; production retains B0's TF32 settings.
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    source = YOLO(str(args.weights)).model.float()
    torch.manual_seed(42)
    baseline = DetectionModel(str(MODELS["b0_o2m"]), nc=4, verbose=False).float()
    baseline.load(source, verbose=False)
    baseline.eval().to(device)
    baseline.args = get_cfg()
    # Include a large target so P4 gets usable regression gradients under fp16.
    sample = torch.randn(2, 3, 256, 256, device=device)
    batch = {"img": sample, "batch_idx": torch.tensor([0., 1.], device=device),
             "cls": torch.zeros(2, 1, device=device),
             "bboxes": torch.tensor([[.5, .5, .1, .08], [.5, .5, .7, .6]], device=device)}
    with torch.no_grad():
        native = baseline(sample)
    raw, decoded = decode_branch(baseline.model[-1], native[1], "o2m")
    expected = xywh2xyxy(native[0][:, :4].transpose(1, 2)).transpose(1, 2)
    torch.testing.assert_close(decoded[:, :4], expected)
    assert branch_gaps([{"model": "B0", "image": "synthetic.jpg", "gt_index": 0,
                         "class": "D00", "branch": "o2m"}]) == ([], [])
    synthetic = {"model": "B0", "branch": "o2m", "image": "synthetic.jpg", "gt_index": 0,
                 "class": "D00", "size": "small", "aspect": "compact_lt2",
                 "global_max_iou": .5, "correct_top100_max_iou": .4,
                 "correct_top1_iou": .3, "final_correct_max_iou": .2}
    paired, _ = operator_gains([synthetic, {**synthetic, "model": "A", "global_max_iou": .6}], "B0")
    assert abs(paired[0]["delta_global_max_iou"] - .1) < 1e-6
    records = []
    for candidate in ("c_o2m", "a_o2m", "r_o2m"):
        torch.manual_seed(42)
        model = DetectionModel(str(MODELS[candidate]), nc=4, verbose=False).float()
        model.load(source, verbose=False)
        model.args = get_cfg()
        model.to(device).eval()
        assert not model.end2end and isinstance(model.init_criterion(), v8DetectionLoss)
        assert model.stride.tolist() == [8., 16., 32.]
        assert isinstance(model.model[-1].strip_cv2[2], torch.nn.Identity)
        assert o2m_state_sha256(model) == o2m_state_sha256(baseline)
        with torch.no_grad():
            output = model(sample)
        for field in ("boxes", "scores"):
            torch.testing.assert_close(output[1][field], native[1][field], atol=0, rtol=0)

        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=.01)
        scaler = torch.amp.GradScaler(device.type, init_scale=128, enabled=device.type == "cuda")
        learned_seen = [False, False]
        offset_seen = [False, False]
        for step in range(3):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                losses, _ = model.loss(batch)
            assert torch.isfinite(losses).all()
            scaler.scale(losses.sum()).backward()
            scaler.unscale_(optimizer)
            for index, adapter in enumerate(model.model[-1].strip_cv2[:2]):
                assert adapter.gamma.grad is not None and torch.isfinite(adapter.gamma.grad).all()
                learned = [p.grad for name, p in adapter.named_parameters() if name != "gamma"]
                assert all(g is None or torch.isfinite(g).all() for g in learned)
                learned_seen[index] |= any(g is not None and g.abs().max() > 0 for g in learned)
                if candidate == "r_o2m":
                    offset_seen[index] |= bool(adapter.offset.weight.grad.abs().max() > 0)
            scaler.step(optimizer)
            scaler.update()
        assert all(learned_seen), (candidate, "no branch gradient", learned_seen)
        if candidate == "r_o2m":
            assert all(offset_seen), (candidate, "no offset gradient", offset_seen)

        model.eval()
        with torch.no_grad():
            result = model(sample)
            backend = AutoBackend(model=make_o2m_model(model), device=device, fuse=True, verbose=False)
            fused = backend(sample)
        torch.testing.assert_close(fused[0], result[0], atol=2e-3, rtol=2e-3)
        assert backend.model.model[-1].strip_cv2 is not None
        validator = O2MValidator(args={"conf": .001, "iou": .7, "max_det": 300, "plots": False},
                                 save_dir=args.report.parent / "validator")
        validator.training, validator.end2end, validator.box_head = True, False, model.model[-1]
        train = validator.postprocess((result[0].clone(), result[1]))
        validator.training = False
        standalone = validator.postprocess((result[0].clone(), result[1]))
        for left, right in zip(train, standalone):
            for field in ("bboxes", "conf", "cls"):
                torch.testing.assert_close(left[field], right[field])
        # Training validation/checkpoint export converts the entire model to half.
        if device.type == "cuda":
            with torch.no_grad():
                half = deepcopy(model).half()(sample.half())
                assert torch.isfinite(half[0]).all()
        checkpoint = args.report.parent / f"{candidate}_engineering.pt"
        torch.save({"model": deepcopy(model).cpu()}, checkpoint)
        restored = torch.load(checkpoint, weights_only=False)["model"].to(device).eval()
        with torch.no_grad():
            torch.testing.assert_close(restored(sample)[0], result[0], atol=0, rtol=0)
        records.append({"candidate": candidate, "passed": True,
                        "params_unfused": sum(p.numel() for p in model.parameters()),
                        "additional_params": sum(p.numel() for p in model.parameters()) - sum(p.numel() for p in baseline.parameters()),
                        "initial_shared_hash": o2m_state_sha256(baseline)})
        print(f"PASS: {candidate} identity, shared weights, gradients, fusion, validation, save/load and AMP")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps({"passed": True, "device": str(device), "records": records}, indent=2))


if __name__ == "__main__":
    main()
