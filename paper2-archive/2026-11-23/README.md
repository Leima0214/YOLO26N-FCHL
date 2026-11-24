# Paper 2 experimental archive

This archival branch is based on c4af5fc04787aad0d12c4fa049739c28c19cbfbe from codex/paper2-operator-screen, the code base identified for the completed B1 + #5 run. Each experiment directory contains the training configuration, 100-epoch records, an independent evaluation summary and a SHA-verified representative checkpoint. The source directory contains the matching experiment and model definitions.

| Directory | Remote run | Representative checkpoint |
|---|---|---|
| cls | HistoricalCLS_DualBest | CLS architecture replay, single checkpoint E79 |
| cls-p2a | HistCLS_SchemeA_P2 | CLS-P2A E90; separate EMA6 evaluation included |
| b1 | b1 | B1 E80, plus the two pre-BN averaged weights for the separate EMA experiments |
| b1-plus-5 | b1_w50 | B1 + #5 E73 and fixed E100 |

The October CLS E77 survives as a historical evaluation record. The archived CLS checkpoint comes from the November replay. For CLS-P2A, the recorded 640 EMA6 evaluation has AP50 70.250%, AP75 45.456% and AP50:95 43.709%. Its six source snapshots have separate recorded hashes. The file cls-p2a/weights/best.pt corresponds to the E90 row below.

## SVRDD7 validation results (%)

| Model | Epoch | Evaluation input | AP50 | AP75 | AP50:95 | AR@100 |
|---|---:|---|---:|---:|---:|---:|
| CLS (replay) | E79 | Uniform640 | 69.789 | 45.385 | 42.984 | 61.262 |
| CLS-P2A | E90 | Uniform640 | 69.633 | 44.953 | 43.234 | 61.265 |
| B1 | E80 | Uniform640 | 69.757 | 46.014 | 43.345 | 61.863 |
| B1 + #5 | E73 | Warp640 | 72.130 | 47.843 | 45.686 | 63.863 |
| B1 + #5 | E100 | Warp640 | 70.846 | 46.559 | 44.558 | 62.279 |
| B1 + #5 | E73 | Warp768 | 73.488 | 49.824 | 46.492 | 64.671 |
| B1 + #5 | E73 | Uniform768 | 70.945 | 46.754 | 44.217 | 63.008 |

B1 + #5 uses the same B1 initialization and 3,246,130 training parameters. It applies the vertical road resampling to the training input and the corresponding evaluation path. Warp evaluations map boxes back to the original image coordinates before class-aware NMS. The 768 rows re-evaluate the same E73 weight at 768. AP50:95 improves by +2.341 percentage points in the 640 best-to-best comparison. The results are SVRDD7 Val results from seed 42 with the designated input recipes.

Use the evaluation JSONs for per-class, scale and slice metrics and for weight identities. The training entry points retain the paths of the original environment. To read archived custom model definitions, use the pinned project code with the source directory on the Python import path. SHA256SUMS lists the checksum of each archived file.
