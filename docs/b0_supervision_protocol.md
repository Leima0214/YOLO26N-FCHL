# Japan4 B0 supervision comparison

T0: verify remote identity, CUDA environment and train/val data integrity. Test stays sealed.
T1: both runs start fresh from the same yolo26n.pt, use seed 42, 640 px, batch 32, workers 8 and 30 total epochs. Explicit MuSGD lr0=0.00125, momentum=0.9, lrf=0.01, warmup_bias_lr=0; remaining arguments are snapshotted by train_paper2.py.
T2: engineering checks and independent 1E smoke; smoke weights are never used to initialize the 30E runs.

Dual B0: fixed 0.8 O2M + 0.2 O2O loss. Pure O2M B0: end2end=False, no O2O branch, O2M loss weight 1.0. This compares supervision recipes, including the different O2M loss weights; it does not isolate the causal effect of adding O2O. The pinned Detect implementation detaches features entering O2O.

Both select best.pt on source Val O2M + class-aware NMS AP50:95 (conf=0.001, IoU=0.7, max_det=300); no Voting/TTA. Independent COCO validation uses maxDets=100, with AP50:95/AP50/AP75, P/R, scale and per-class metrics. Preserve native and COCO values separately. Record effective optimizer groups and the hash of all initialized common tensors at on_train_start.

Order: dual B0 30E, then pure O2M B0 30E, then unified Val evaluation and comparison. Stop the queue on a failed stage. Single-seed, short-schedule results are exploratory, not evidence of seed robustness or final 100E performance.
