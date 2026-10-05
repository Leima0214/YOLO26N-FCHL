# Paper 2 operator screen, 2026-10-05

This is a single-seed 30E feasibility screen. It tests whether regression refinements improve localization, and whether directional and deformable operators help different instances. It does not establish a morphology mechanism or justify a router yet.

## Frozen comparison

- Data: Japan4-cleanV3 train 6320 / Val 790 images; class order D00, D10, D20, D40. Test remains sealed. No target-domain data informs selection.
- B0: completed pure O2M `B0_O2M_30E_seed42_20261004`, best checkpoint selected by training Val O2M AP50:95. This existing run is reused after checking initialization and effective optimizer against new runs.
- C: three ordinary depthwise 3x3 branches in the same static softmax-fusion / zero-gamma residual shell as A.
- A: existing static Strip refinement: 3x3, 1x7, 7x1 on P3; 3x3, 1x5, 5x1 on P4. Fusion weights are global learned scalars, not an input-dependent router.
- R: one depthwise deformable 3x3 branch with a learned 18-channel offset convolution and zero-gamma residual. Offset groups are shared across channels; no modulation mask or router.
- All refinements affect only P3/P4 regression inputs. P5 and classification inputs remain native. Failure therefore does not disprove deformable localization at other feature levels.
- Additional unfused parameters versus B0: C 5192, A 3912, R 32870. C is a capacity control near A, not an exact parameter/FLOP match for R. C's three linear 3x3 branches can be merged mathematically; this screen retains the same shell as A.
- Fresh official `weights/yolo26n.pt` for each 1E smoke and each independent 30E run; no smoke-to-30E continuation. Official SHA256: `9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef`.
- Pure O2M, loss weight 1.0, `v8DetectionLoss`, O2M + class-aware NMS; seed42, 640, batch32, workers8, 30E. MuSGD/AMP/deterministic setting, LR/warmup/augmentation and checkpoint selection inherit `TRAINING_PROTOCOL` from the completed B0. Effective optimizer and initialized shared-state hashes must match B0 before formal screening starts.

## Engineering and evidence

Engineering checks cover identity initialization, shared tensors, gradient flow (including R offsets), CUDA AMP, half-precision validation, fusion, save/load, and pure-O2M decoding. Independent 1E smokes check the actual dataset/trainer; their AP is not scientific evidence.

R's torchvision CUDA deformable backward reports a nondeterministic implementation under PyTorch's `warn_only=True` deterministic setting. The common seed/protocol is preserved, but bitwise repeatability is not claimed. Fusion equivalence is checked with TF32 disabled in the engineering process only; production retains B0's arithmetic settings.

Before training, B0 Val diagnostics record unfiltered best-IoU candidate, its P3/P4/P5 level, score-ranked candidates and final NMS candidate coverage. A deterministic 200-GT class/size sample is prepared for human morphology review. The morphology fields remain blank until a person examines the images. Class ID and HBB aspect ratio do not establish linear/network/region morphology.

After sequential C -> A -> R 30E, the queue evaluates all four selected checkpoints with identical standalone Val settings: batch32, rect, conf0.001, NMS IoU0.7, validator max_det300; COCO maxDets100. Native training and standalone fused saved-checkpoint metrics are reported separately. COCO AP50:95 is the primary comparison; AP75, per-class AP, small/medium/large AP, recall, parameters and latency expose tradeoffs. R's THOP estimate excludes deformable sampling/interpolation cost; it is not a complete FLOP count.

Paired per-GT deltas compare raw best IoU, same-class top100/top1 and final candidate coverage with B0 on identical GT keys. These diagnostic bounds are not deployable AP or official COCO recall. A useful operator must show a localization benefit against B0/C with tolerable class/size regressions; shape-specific claims require human-validated morphology and subsequent intervention/repeat experiments. No threshold or operator choice is changed after examining this screen's results.

The runner stops on any failed stage, preserves logs, and never overwrites previous runs. Once launched it needs no interactive session. No monitoring automation is created.
