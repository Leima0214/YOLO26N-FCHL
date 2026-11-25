# PDS-YOLO & Paper 2 Complete Experimental Archive (SVRDD7)

This archive packages the complete verified code, configurations, 100-epoch training logs, independent SVRDD7 Val evaluation summaries, RTX 4090 D efficiency benchmarks, 34k candidate diagnostics, and SHA-verified checkpoints (`best.pt` + `last.pt`) for **PDS-YOLO (`b1-plus-5` / `b1_w50`)**, its full progressive ablation chain (**`B0` -> `B0+P2` -> `CLS` -> `CLS-P2A` -> `B1` -> `B1+#5 / PDS-YOLO`** plus `B0+#5`, `CLS-P2A+#5`, and `B1@768`), and completed official YOLO baselines (**`YOLOv8n`**, **`YOLOv9t`**).

- **Base commit**: `codex/paper2-operator-screen` (`c4af5fc04787aad0d12c4fa049739c28c19cbfbe`) & `archive/paper2-20261123` (`c0f05a7a2413febede9fd50224acb6c805351c8f`)
- **Dataset split discipline**: All models are trained on `SVRDD7 Train` (6,000 images, 15,619 GT) and evaluated on `SVRDD7 Val` (1,000 images, 2,612 GT) with `seed=42`, FP32 inference, O2M + class-aware NMS (`iou=0.70`, except O2O rows), COCO `maxDets=100`. `SVRDD7 Test` was never accessed.
- **PDS-YOLO design summary**: Combines High-Resolution Lesion head (`HRL / P2`), Strip/Asymmetric Depthwise Convolution (`SDC / P2A`), Re-parameterized Shallow Dense Edge Enhancement (`HDE / B1`), and Vertical Pavement Perspective Resampling (`PPR / #5`, piecewise warp `256/384` at `640x640`).

## Directory Map

| Directory | Role / Run ID | Notes |
|---|---|---|
| `b0` | Native YOLO26n (`b0`) | Official O2M+O2O YOLO26n trained at 640 uniform (O2M primary + O2O & 768 eval) |
| `b0-p2` | YOLO26n + HRL (`b0_p2`) | YOLO26n + P2 fusion without PKI2/SDC/HDE |
| `b0-plus-5` | YOLO26n + PPR (`b0_w50`) | Native YOLO26n with `#5` vertical pavement resampling (Warp 256/384) |
| `cls` | Historical CLS replay (`HistoricalCLS_DualBest`) | P2 + PKI2 baseline replay (`best.pt` at E79) |
| `cls-p2a` | HRL + SDC (`HistCLS_SchemeA_P2`) | Asymmetric P2 fusion (`best.pt` at E90 + EMA6 summary) |
| `cls-p2a-plus-5` | HRL + SDC + PPR (`p2a_w50`) | `CLS-P2A` trained & evaluated with `#5` Warp 256/384 |
| `b1` | HRL + SDC + HDE (`b1`) | Full PDS network at 640 uniform + EMA6/BN recalibration weights |
| **`b1-plus-5`** | **PDS-YOLO (`b1_w50`)** | **Final promoted model (`best.pt` E73 + `last.pt` E100 + 34k diagnosis)** |
| `b1-768` | High-res Compute Control (`b1_768`) | `B1` trained at 768 uniform (`1.44x` pixels, `13.202 GFLOPs`) |
| `baselines/v8n` | Official `YOLOv8n` (`v8n`) | COCO80-pretrained, `AdamW(lr0=0.000909)`, 640 100E |
| `baselines/v9t` | Official `YOLOv9t` (`v9t`) | COCO80-pretrained, `AdamW(lr0=0.000909)`, 640 100E |
| `source/` | Complete experiment & model scripts | `candidates*.py`, `run*.py`, `diagnose.py`, `bench_and_diag_v7.py`, `run_official_yolo.py`, etc. |
| `diagnostics/` | Verified diagnostic & benchmark JSONs | `v7_latency_and_flops.json`, `v7_nms_coord_audit.json`, `b1_w50_34k_diagnosis.json`, etc. |

## SVRDD7 Val COCO Results (%)

| Model | Ckpt (Epoch) | Input | `AP50` | `AP75` | `AP50:95` | `AR100` | `AP_small` | `AP_medium` | `AP_large` |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| B0 (`YOLO26n` O2M) | best (`E74`) | Uniform640 | 67.793 | 42.661 | 41.220 | 60.386 | 19.048 | 35.203 | 55.343 |
| B0 (`YOLO26n` O2O) | o2o_best (`E74`) | Uniform640_O2O | 64.557 | 39.990 | 38.541 | 60.665 | 18.025 | 32.005 | 54.966 |
| B0 + P2 (`HRL`) | best (`E83`) | Uniform640 | 69.329 | 45.289 | 43.099 | 61.815 | 21.810 | 36.576 | 59.820 |
| B0 + #5 (`PPR` O2M) | best (`E82`) | Warp640 | 70.721 | 45.850 | 44.240 | 61.579 | 21.224 | 38.928 | 57.955 |
| CLS (replay) | best (`E79`) | Uniform640 | 69.789 | 45.385 | 42.984 | 61.262 | 21.428 | 36.149 | 58.340 |
| CLS-P2A (`HRL+SDC`) | best (`E90`) | Uniform640 | 69.633 | 44.953 | 43.234 | 61.265 | 21.512 | 35.985 | 58.781 |
| CLS-P2A + #5 | best (`E76`) | Warp640 | 70.826 | 47.569 | 45.096 | 63.264 | 23.509 | 37.514 | 61.456 |
| B1 (`HRL+SDC+HDE`) | best (`E80`) | Uniform640 | 69.757 | 46.014 | 43.345 | 61.863 | 21.351 | 36.252 | 62.411 |
| **PDS-YOLO (`B1 + #5`)** | best (`E73`) | Warp640 | 72.130 | 47.843 | 45.686 | 63.863 | 23.478 | 38.628 | 58.761 |
| PDS-YOLO (`last`) | last (`E100`) | Warp640 | 70.846 | 46.559 | 44.558 | 62.279 | 23.759 | 37.962 | 58.332 |
| B1 @ 768 (Compute Control) | best (`E70`) | Uniform768 | 71.765 | 48.514 | 45.450 | 63.929 | 24.558 | 36.945 | 57.483 |
| B1 @ 768 (`best_ap50`) | best_ap50 (`E60`) | Uniform768 | 72.696 | 48.237 | 45.270 | 64.154 | 24.847 | 36.760 | 55.933 |
| YOLOv8n (official) | best (`E85`) | Uniform640 | 66.273 | 40.515 | 39.762 | 58.940 | 17.518 | 34.513 | 54.994 |
| YOLOv9t (official) | best (`E89`) | Uniform640 | 69.137 | 43.579 | 41.876 | 59.862 | 17.853 | 35.303 | 62.132 |

## RTX 4090 D Efficiency & End-to-End Speed (200 SVRDD7 Val `1024x1024` images)

> Deployed weights (`BaseModel.fuse()`, O2M drops O2O branch, `HDE` folds into single `3x3` conv). End-to-end `bs=1` includes CPU resize/warp -> H2D -> FP32 forward -> coordinate inverse mapping -> NMS (`conf=0.25, iou=0.70`) -> D2H.

| Model | `imgsz` | Train / Deploy Params | `GFLOPs` | GPU FP32 `bs=32/1` (ms) | GPU FP16 `bs=32/1` (ms) | E2E `bs=1` (ms) | E2E `FPS` |
|---|---:|---:|---:|---:|---:|---:|---:|
| B0 (`YOLO26n` O2O) | 640 | 2,506,530 / 2,376,201 | 5.196 | 0.483 / 8.511 | 0.290 / 8.999 | 10.931 | 91.48 |
| B0 (`YOLO26n` O2M) | 640 | 2,506,530 / 2,376,201 | 5.196 | 0.473 / 8.081 | 0.276 / 8.649 | 11.169 | 89.53 |
| B0 + #5 (`PPR`) | 640 | 2,506,530 / 2,376,201 | 5.196 | 0.473 / 8.110 | 0.277 / 8.587 | 11.973 | 83.52 |
| B0 + P2 (`HRL`) | 640 | 2,574,360 / 2,421,604 | 7.496 | 0.894 / 8.905 | 0.496 / 9.280 | 11.995 | 83.37 |
| CLS (`HistCLS`) | 640 | 3,195,674 / 3,042,918 | 8.340 | 1.042 / 11.565 | 0.603 / 12.354 | 15.467 | 64.65 |
| CLS-P2A (`HRL+SDC`) | 640 | 3,208,050 / 3,054,370 | 8.487 | 1.096 / 11.964 | 0.629 / 12.902 | 15.089 | 66.27 |
| B1 (`HRL+SDC+HDE`) | 640 | 3,246,130 / 3,067,810 | 9.168 | 1.209 / 12.254 | 0.686 / 13.099 | 15.398 | 64.95 |
| CLS-P2A + #5 | 640 | 3,208,050 / 3,054,370 | 8.487 | 1.096 / 12.037 | 0.628 / 12.806 | 15.940 | 62.73 |
| **PDS-YOLO (`b1_w50`)** | 640 | 3,246,130 / 3,067,810 | 9.168 | 1.209 / 12.237 | 0.686 / 13.007 | 16.484 | 60.66 |
| B1 @ 768 (Compute Control) | 768 | 3,246,130 / 3,067,810 | 13.202 | 1.810 / 12.166 | 1.020 / 12.977 | 15.889 | 62.94 |
| YOLOv8n (official) | 640 | 3,012,213 / 3,007,013 | 8.092 | 0.451 / 5.077 | 0.249 / 5.147 | 8.046 | 124.28 |
| YOLOv9t (official) | 640 | 2,006,773 / 1,972,741 | 7.611 | 0.577 / 14.038 | 0.471 / 14.623 | 17.616 | 56.77 |

## Integrity & Verification

```bash
cd paper2-archive/2026-11-23
sha256sum -c SHA256SUMS
```

See `metrics.csv` for machine-readable evaluation metrics across all checkpoints, `source/` for all model definitions and evaluation scripts, and `diagnostics/` for NMS coordinate audits (`v7_nms_coord_audit.json`) and 34k candidate recall/FP decompositions (`b1_w50_34k_diagnosis.json`).
