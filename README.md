# YOLO26N-FCHL

Paper 2 道路损伤定位研究的起点：在同一 YOLO26n 基线上，检验不同几何算子是否适合不同形态，再决定是否需要由特征决定融合权重。

本仓库当前提供 **B0 基线、已有静态 Strip 候选和审计工具**。FCHL 是仓库名称，尚不代表已经实现或验证了完整的特征条件定位方法。变形卷积候选与动态融合应在前置分析和对照实验支持后再实现。

## 来源与迁移范围

代码来自 [YOLO26N-RDP4 的指定分支](https://github.com/Leima0214/YOLO26N-RDP4/tree/codex/rgt-gbrg-r10v-t1)，固定提交 `eb6e250d49204631e84f180a4f78f64a968c6a28`，没有混入本机其他分支或未提交修改。逐文件来源、SHA256 和修改标记见 [migration_manifest.json](docs/migration_manifest.json)。许可证沿用原仓库 AGPL-3.0。

| 文件或目录 | 用途 |
| --- | --- |
| `ultralytics/` | 原分支运行核心；保留导入和检查点兼容所需模块 |
| `ultralytics/cfg/models/26/yolo26.yaml` | YOLO26n 原生 B0 |
| `ultralytics/cfg/models/26/yolo26n-japan4-s1-strip-regression.yaml` | S1：P3/P4 回归特征上的静态 Strip 残差 |
| `scripts/train_paper2.py` | B0/S1 训练入口、固定双头权重、显式 O2M 验证和运行记录 |
| `scripts/eval_paper2.py` | Common4 验证集上的 O2M+NMS 与 COCO 指标 |
| `scripts/audit_japan4_dataset.py` | 现有 Japan4-cleanV3 的 train/val 和分组清单审计 |
| `scripts/diagnose_japan4_head_candidates.py` | O2M/O2O 原始候选框、排名、尺寸与长宽比诊断 |
| `scripts/verify_japan4_s1.py` | 继承权重、零初始化、梯度、融合检查；可选 ONNX |
| `scripts/compare_japan4_runs.py` | 训练曲线比较 |
| `tests/check_migration.py` | 随机权重和人工张量的迁移完整性检查 |

原运行核心的 `tasks.py` 会导入较多历史模块，因此这里保留了兼容代码。Paper 2 入口仅开放 B0/S1，不启用 RoadSnake、GBRG、MoE 或 Mamba 等历史方法。旧实验配置、数据、训练权重、结果表及远程机器设置没有迁移；Ultralytics 自带示例图和数据集 YAML 属于运行包资源。

S1 的横向、纵向和方形分支使用全局可学习权重及零初始化残差。它**不是**逐目标或逐位置的形态路由，也没有通过类别标签给样本指定算子。

## 安装

推荐在独立 Python 3.11+ 环境中安装与 CUDA 相匹配的 PyTorch/torchvision，再安装本仓库：

```bash
git clone https://github.com/Leima0214/YOLO26N-FCHL.git
cd YOLO26N-FCHL
python -m pip install -e .
python tests/check_migration.py
```

运行仓库脚本时应使用此环境，避免其他 Ultralytics 安装覆盖本地模块。`docs/source_requirements_roadlite26.txt` 保留原分支的历史依赖锁定记录；当前安装依赖见 `requirements.txt`，正式实验还须保存实际环境版本。

本次 Windows 工程检查使用 Python 3.12.7、torch 2.12.0，并在局部 `.venv` 中补充缺失依赖。继承 Anaconda 库时出现重复 OpenMP 运行库冲突，检查进程暂时使用 `KMP_DUPLICATE_LIB_OK=TRUE`；这不是正式训练环境的推荐配置，也没有写入代码或全局设置。正式实验应使用独立、无此冲突的环境。

## 数据与实验边界

复制 `configs/datasets/japan4.example.yaml` 为被 Git 忽略的 `japan4.local.yaml`，填入现有数据集的绝对路径。保留已有划分、四类顺序和分组清单；数据审计需要数据根目录下的 `audit/split_assignment.csv` 与 `audit/split_report.json`。独立 COCO 评估还需要 `annotations/instances_val.json`。

模板不列出 Test。Czech4 模板仅用于源域方案冻结后的目标域验证；任何会影响模型选择的 Czech 检查都不符合严格的目标域未参与选择设定。训练入口检查四类顺序，**不能凭类别名称自动区分 Japan 与 Czech**，执行前须核对数据根目录与划分。

先验证长宽比等代理指标与实际形态的关系，再分析失败样本。D00/D10/D20/D40 是损伤类别，不是几何形态标签。原始候选诊断的 oracle 统计已更名为 `oracle_assigned_recall_mean_iou50_95`：它是 IoU 阈值上的匹配召回统计，不能写成检测 AP。

## 入口示例

以下仅为以后实验的命令，本次迁移没有执行它们。权重与数据须自行放在本机或服务器上。

```bash
python scripts/audit_japan4_dataset.py --data configs/datasets/japan4.local.yaml --output reports/japan4_audit.json

python scripts/train_paper2.py --candidate b0 --data configs/datasets/japan4.local.yaml --epochs 30 --weights weights/yolo26n.pt --batch 32 --device 0 --seed 42 --name b0_30e_seed42
python scripts/train_paper2.py --candidate s1 --data configs/datasets/japan4.local.yaml --epochs 30 --weights weights/yolo26n.pt --batch 32 --device 0 --seed 42 --name s1_30e_seed42

python scripts/eval_paper2.py --checkpoint B0=runs/paper2/b0_30e_seed42/weights/best.pt --checkpoint S1=runs/paper2/s1_30e_seed42/weights/best.pt --data configs/datasets/japan4.local.yaml --output reports/source_comparison --device 0

python scripts/diagnose_japan4_head_candidates.py --model B0=runs/paper2/b0_30e_seed42/weights/best.pt --data configs/datasets/japan4.local.yaml --output reports/raw_candidates --skip-val-sweep --device 0
```

训练入口固定 O2M/O2O 损失系数为 `0.8/0.2`，按 **源域验证集的 O2M+类别内 NMS AP50:95** 选择 `best.pt`。这是新入口明确的协议，不等同于原框架逐轮衰减的默认权重；直接调用 `YOLO.train()` 不会自动启用该协议。训练期保留双头输出计算损失，指标单独解码 O2M；独立评估先复制 O2M 模型，再融合卷积，避免原生 O2O 融合删除 O2M 分支。仅保留 O2O 的部署权重会被拒绝。

`optimizer=auto` 沿用源入口，因此不同参数量可能影响有效学习率；运行记录包含实际优化器和参数组。结构因果比较还应核对或固定有效优化器与学习率，不能仅根据 `args.yaml` 宣称完全匹配。初始化权重哈希、模型/数据配置快照、Git 状态与环境记录写入 `runtime_meta/`。中断恢复须使用同一运行的 `last.pt` 和原总轮次，不能将 30E 延长成正式 100E。

独立评估只开放 `val`，没有 Box Voting。COCO AP/AR 的最大检测数为 100，验证器输出上限为 300；原生验证指标与 COCO 重算指标不可混用。GPU 延迟为 PyTorch 单批次前向延迟，未包含 NMS 与数据处理。

本次已验证模型构建、双头零初始化等价、固定损失权重、O2M 选择和融合、空检测 COCO 处理及文件来源哈希。尚未验证真实数据训练、恢复、多 GPU、ONNX 导出或检测精度；没有产生新的科研结果。
