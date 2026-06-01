# 基于 YOLOv8 的水下鱼类目标检测与数量统计

## 一、项目简介

本项目是一个基于 **YOLOv8** 的水下鱼类目标检测与数量统计系统，作为《计算机视觉》课程设计项目。

项目以水下视频序列为数据源，将 MOT 格式的 `gt.txt` 标注转换为 YOLO 训练格式，训练鱼类检测模型，并在独立测试序列上进行检测、计数、量化评估和视频可视化。

**核心技术点**：

- 目标检测任务（YOLOv8）
- 数据格式转换（MOT → YOLO）
- 深度学习模型训练与迁移学习
- 检测结果可视化（OpenCV）
- 鱼类数量统计
- 跨视频序列独立测试与量化评估

---

## 二、项目功能

- **数据格式转换**：将 MOT 格式的 `gt.txt` 标注文件转换为 YOLO 训练格式，支持抽帧、限制帧数、清空旧输出
- **YOLOv8 模型训练**：基于 YOLOv8 预训练权重进行迁移学习，支持自定义 epochs、imgsz、batch 等参数
- **图片/视频鱼类检测**：对单张图片、图片文件夹或视频进行目标检测
- **鱼数量统计**：逐帧统计鱼类数量，输出带检测框和数量标注的结果图片/视频，生成 `fish_count.csv`
- **量化评估**：基于独立测试序列的 gt.txt 进行逐帧评估，输出 Precision、Recall、F1-score、Mean IoU、Count MAE、Count RMSE、Count Accuracy
- **评估图表生成**：自动生成数量对比曲线、数量误差图、指标柱状图
- **帧合成视频**：将图片帧合成为 mp4 视频（支持原始帧和检测结果帧）

---

## 三、项目结构

```
FishCount/
├── data/
│   ├── BT-001/                          # 训练序列
│   │   ├── img1/                        # 图片帧（000001.jpg ~ ...）
│   │   └── gt/
│   │       └── gt.txt                   # MOT 格式标注
│   └── BT-003/                          # 独立测试序列（不参与训练）
│       ├── img1/
│       └── gt/
│           └── gt.txt
├── yolo/
│   ├── gt_to_yolo.py                    # gt.txt → YOLO 数据集转换
│   ├── train_yolo.py                    # YOLOv8 模型训练
│   ├── predict_yolo.py                  # YOLOv8 图片/视频预测
│   ├── count_fish.py                    # 鱼数量统计与可视化
│   ├── evaluate_yolo.py                 # 基于 gt.txt 的量化评估
│   └── frames_to_video.py               # 图片帧合成视频
├── output/                              # 输出目录（自动生成，已 gitignore）
│   ├── yolo_dataset_BT001/              # BT-001 转换的 YOLO 数据集
│   ├── yolo_runs/                       # 训练与预测结果
│   ├── eval_results_BT003/              # BT-003 评估结果
│   ├── count_results_BT003/             # BT-003 计数结果
│   └── videos/                          # 合成视频
├── pixi.toml                            # 项目依赖配置（pixi + conda-forge）
├── .gitignore
└── README.md
```

---

## 四、环境依赖

本项目使用 **pixi**（基于 conda-forge）管理 Python 环境与依赖，Python 版本 **3.13+**。

**主要依赖**：

| 依赖 | 用途 |
|------|------|
| `ultralytics` | YOLOv8 训练与推理 |
| `pytorch` | 深度学习框架 |
| `torchvision` | PyTorch 视觉工具集 |
| `opencv` | 视频读写与检测框绘制 |
| `pillow` | 图像读取 |
| `matplotlib` | 评估图表绘制 |
| `numpy` | 数值计算（pytorch 自动带入） |

**安装方式**：

```bash
# 安装 pixi（如未安装）
# Windows: winget install prefix-dev.pixi
# 参考: https://pixi.sh/latest/#installation

# 安装全部依赖
pixi install
```

所有命令均通过 `pixi run python` 执行，或在 VS Code 中将解释器切换为 `.pixi\envs\default\python.exe`。

> 如果使用 pip 管理依赖，可执行：
> ```bash
> pip install ultralytics opencv-python pillow matplotlib
> ```
> （numpy 和 pytorch 会随 ultralytics 自动安装）

---

## 五、数据集说明

### gt.txt 格式

`gt.txt` 为类 MOT 格式，每行一个标注框，字段以逗号或空格分隔：

```
frame_id, object_id, x, y, w, h, confidence, ...
```

| 字段 | 含义 |
|------|------|
| `frame_id` | 帧编号，对应图片文件名中的数字 |
| `object_id` | 目标（鱼）编号 |
| `x, y` | 目标框左上角坐标（像素） |
| `w, h` | 目标框宽度和高度（像素） |
| `confidence` | 标注置信度（可选，≤0 视为无效） |

### YOLO 标签格式

转换后的标签为每行一个目标：

```
class_id center_x center_y width height
```

其中 `center_x`、`center_y`、`width`、`height` 均归一化到 [0, 1] 区间。

---

## 六、实验设计

本项目采用 **跨视频序列测试** 方式：

- **BT-001**：训练序列，用于生成 YOLO 数据集，随机划分 80% 训练集、20% 验证集
- **BT-003**：独立测试序列，**不参与训练**，仅用于最终评估和可视化

**设计原因**：同一视频的连续帧高度相似，若训练集和测试集来自同一序列，评估结果会偏乐观。使用 BT-003 作为独立测试集，能更真实地反映模型在新视频场景中的泛化能力。

**完整流程**：

```
BT-001 gt.txt + img1  →  gt_to_yolo.py  →  YOLO 数据集  →  train_yolo.py  →  模型 best.pt (fish_BT001_yolov8n_640)
                                                                                    ↓
BT-003 img1  →  count_fish.py  →  检测计数结果 + fish_count.csv
BT-003 gt.txt + img1  →  evaluate_yolo.py  →  量化评估 + 图表
BT-003 img1  →  frames_to_video.py  →  原始视频
检测结果帧    →  frames_to_video.py  →  可视化视频
```

---

## 七、使用方法

以下命令均为一行格式，路径使用 `/`，在项目根目录下执行。

### 1. 用 BT-001 生成 YOLO 数据集

```bash
python yolo/gt_to_yolo.py --gt data/BT-001/gt/gt.txt --img-dir data/BT-001/img1 --output output/yolo_dataset_BT001 --val-ratio 0.2 --frame-step 5 --max-frames 600 --clean
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--gt` | 必填 | gt.txt 标注文件路径 |
| `--img-dir` | 必填 | 图片帧目录 |
| `--output` | `output/yolo_dataset_sampled` | 输出目录 |
| `--val-ratio` | `0.2` | 验证集比例 |
| `--frame-step` | `1` | 抽帧间隔（每隔 N 帧取 1 帧） |
| `--max-frames` | 不限 | 最多使用帧数 |
| `--seed` | `42` | 随机种子 |
| `--class-name` | `fish` | 类别名称 |
| `--split-mode` | `random` | 划分方式：`random` / `sequential` |
| `--filter-conf` | 关闭 | 过滤 conf≤0 的标注 |
| `--clean` | 关闭 | 转换前清空输出目录 |

### 2. 快速验证训练（推荐先跑）

```bash
python yolo/train_yolo.py --data output/yolo_dataset_BT001/data.yaml --model yolov8n.pt --epochs 5 --imgsz 320 --batch 4 --project output/yolo_runs --name fish_BT001_quick --exist-ok
```

### 3. 正式训练

```bash
python yolo/train_yolo.py --data output/yolo_dataset_BT001/data.yaml --model yolov8n.pt --epochs 50 --imgsz 640 --batch 4 --project output/yolo_runs --name fish_BT001_yolov8n_640 --patience 20 --exist-ok
```

如有 NVIDIA 显卡且安装了 CUDA 版 PyTorch，可加 `--device 0`。

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data` | 必填 | data.yaml 路径 |
| `--model` | `yolov8n.pt` | 预训练模型（yolov8n/s/m.pt） |
| `--epochs` | `20` | 训练轮数 |
| `--imgsz` | `416` | 输入图片尺寸 |
| `--batch` | `8` | 批次大小（显存不足时减小） |
| `--workers` | `4` | 数据加载线程数 |
| `--device` | 自动 | 训练设备（`0` = GPU 0，`cpu` = CPU） |
| `--project` | `output/yolo_runs` | 输出根目录 |
| `--name` | `fish_yolov8n` | 训练名称 |
| `--patience` | `10` | 早停耐心值 |
| `--exist-ok` | 关闭 | 允许覆盖同名输出 |
| `--cache` | 关闭 | 缓存图片加速训练 |

### 4. 使用 BT-003 进行量化评估

```bash
python yolo/evaluate_yolo.py --weights output/yolo_runs/fish_BT001_yolov8n_640/weights/best.pt --gt data/BT-003/gt/gt.txt --img-dir data/BT-003/img1 --output output/eval_results_BT003_n640 --imgsz 640 --conf 0.10 --eval-iou 0.5 --frame-step 5 --max-frames 200
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--weights` | 必填 | 模型权重路径 |
| `--gt` | 必填 | gt.txt 标注文件路径 |
| `--img-dir` | 必填 | 图片帧目录 |
| `--output` | `output/eval_results` | 输出目录 |
| `--imgsz` | `416` | 预测输入尺寸 |
| `--conf` | `0.25` | 置信度阈值 |
| `--nms-iou` | `0.45` | YOLO NMS IOU 阈值 |
| `--eval-iou` | `0.5` | 判定 TP 的 IoU 阈值 |
| `--device` | 自动 | 预测设备 |
| `--class-id` | `0` | 目标类别（0=fish，-1=全部） |
| `--frame-step` | `1` | 评估帧间隔 |
| `--max-frames` | 不限 | 最多评估帧数 |
| `--filter-conf` | 关闭 | 过滤 gt 中 conf≤0 的标注 |

### 5. 使用 BT-003 生成检测计数结果

```bash
python yolo/count_fish.py --weights output/yolo_runs/fish_BT001_yolov8n_640/weights/best.pt --source data/BT-003/img1 --output output/count_results_BT003_n640 --imgsz 640 --conf 0.10
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--weights` | 必填 | 模型权重路径 |
| `--source` | 必填 | 输入源（图片/文件夹/视频） |
| `--output` | `output/count_results` | 输出目录 |
| `--imgsz` | `416` | 预测输入尺寸 |
| `--conf` | `0.25` | 置信度阈值 |
| `--iou` | `0.45` | NMS IOU 阈值 |
| `--device` | 自动 | 预测设备 |
| `--class-id` | `0` | 目标类别（0=fish，-1=全部） |

### 6. 将检测结果帧合成为视频

```bash
python yolo/frames_to_video.py --input-dir output/count_results_BT003_n640/images --output output/videos/BT003_fish_count_n640.mp4 --fps 25
```

### 7. 将 BT-003 原始帧合成为视频

```bash
python yolo/frames_to_video.py --input-dir data/BT-003/img1 --output output/videos/BT003_origin.mp4 --fps 25
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--input-dir` | 必填 | 图片帧目录 |
| `--output` | 必填 | 输出视频路径（.mp4） |
| `--fps` | `25` | 视频帧率 |
| `--resize-width` | 不限 | 可选缩放宽度 |
| `--resize-height` | 不限 | 可选缩放高度 |

> 也可使用 pixi 快捷任务：`pixi run dataset`、`pixi run train`、`pixi run evaluate`、`pixi run count`、`pixi run tran`、`pixi run predict`，具体参数见 `pixi.toml` 中 `[tasks]` 段。

---

## 八、输出结果说明

| 路径 | 说明 |
|------|------|
| `output/yolo_dataset_BT001/` | BT-001 转换得到的 YOLO 格式数据集 |
| `output/yolo_dataset_BT001/data.yaml` | 数据集配置文件，含 train/val 路径和类别映射 |
| `output/yolo_runs/fish_BT001_yolov8n_640/` | 正式训练结果（含日志、指标图、权重） |
| `output/yolo_runs/fish_BT001_yolov8n_640/weights/best.pt` | 训练最佳模型 |
| `output/yolo_runs/fish_BT001_yolov8n_640/weights/last.pt` | 最后一轮模型 |
| `output/eval_results_BT003_n640/` | BT-003 独立测试评估结果（imgsz=640） |
| `output/eval_results_BT003_n640/frame_metrics.csv` | 逐帧评估明细 |
| `output/eval_results_BT003_n640/metrics_summary.json` | 总体评估指标 JSON |
| `output/eval_results_BT003_n640/count_curve.png` | 真实数量 vs 预测数量对比曲线 |
| `output/eval_results_BT003_n640/count_error.png` | 逐帧数量误差图 |
| `output/eval_results_BT003_n640/metrics_bar.png` | Precision / Recall / F1 指标柱状图 |
| `output/count_results_BT003_n640/fish_count.csv` | 鱼数量统计 CSV |
| `output/count_results_BT003_n640/images/` | 带检测框和 Fish Count 标注的结果图片 |
| `output/videos/BT003_fish_count_n640.mp4` | 检测计数结果合成视频 |
| `output/videos/BT003_origin.mp4` | BT-003 原始帧合成视频 |

---

## 九、评估指标说明

评估时，将模型预测框与 BT-003 的 gt.txt 标注框逐一比对。当预测框与真实框的 **IoU > `--eval-iou`**（默认 0.5）时，认为该预测正确匹配。

**核心指标**：

| 指标 | 公式 | 说明 |
|------|------|------|
| **TP** | — | 预测框与真实框 IoU > 阈值，匹配成功 |
| **FP** | — | 预测框未能与任何真实框匹配（误检） |
| **FN** | — | 真实框未被任何预测框匹配（漏检） |
| **Precision** | TP / (TP + FP) | 检测出的目标中有多少是真正的鱼 |
| **Recall** | TP / (TP + FN) | 真实存在的鱼中有多少被检测出来 |
| **F1-score** | 2 × P × R / (P + R) | Precision 与 Recall 的综合指标 |
| **Mean IoU** | ΣIoU / TP | 匹配成功的预测框与真实框的平均重叠程度 |
| **Count MAE** | Σ\|pred − gt\| / N | 逐帧鱼数量的平均绝对误差 |
| **Count RMSE** | sqrt(Σ(pred − gt)² / N) | 数量统计的均方根误差 |
| **Count Accuracy** | 1 − Σ\|pred − gt\| / Σgt | 基于数量误差的计数准确率 |

---

## 十、CSV 字段说明

### fish_count.csv（count_fish.py 输出）

| 字段 | 类型 | 说明 |
|------|------|------|
| `frame_id` | int | 图片序号，从 1 开始 |
| `file_name` | str | 图片或视频文件名 |
| `fish_count` | int | 当前帧检测到的鱼数量 |
| `avg_conf` | float | 当前帧所有检测框的平均置信度 |

### frame_metrics.csv（evaluate_yolo.py 输出）

| 字段 | 类型 | 说明 |
|------|------|------|
| `frame_id` | int | 评估帧编号 |
| `file_name` | str | 图片文件名 |
| `gt_count` | int | 真实鱼数量（来自 gt.txt） |
| `pred_count` | int | 模型预测鱼数量 |
| `count_error` | int | 预测数量 − 真实数量 |
| `tp` | int | 正确检测数 |
| `fp` | int | 误检数 |
| `fn` | int | 漏检数 |
| `precision` | float | 当前帧 Precision |
| `recall` | float | 当前帧 Recall |
| `f1` | float | 当前帧 F1-score |
| `mean_iou` | float | 当前帧匹配框平均 IoU |

---

## 十一、常见问题

**1. 训练很慢怎么办？**

- 减小 `--epochs`（如 5）
- 减小 `--imgsz`（如 320）
- 减小 `--max-frames`（如 100）
- 增大 `--frame-step`（如 10）
- 如有 GPU，使用 `--device 0`
- 开启 `--cache` 缓存图片

**2. 检测结果太少（漏检多）怎么办？**

可以降低置信度阈值：

```bash
--conf 0.10
```

极端情况下可尝试 `--conf 0.05`，但会增加误检。

**3. 为什么要使用 `--clean`？**

旧数据集的图片和标签可能残留。例如上次跑了全部 3000 帧，这次设了 `--frame-step 10`，如果不 `--clean`，旧的全量标签仍在，训练会读到错误数据。开启 `--clean` 能保证输出目录完全来自本次转换。

**4. 为什么使用 BT-003 做独立测试集？**

BT-003 未参与训练，其水下场景、光照、鱼群密度可能与 BT-001 不同。用它评估能更真实地反映模型的泛化能力，避免"同一视频切分训练/测试"带来的虚假高指标。

**5. 评估时的 `--conf` 为什么设 0.10？**

评估脚本默认 `--conf` 为 0.25，但水下场景中鱼体较小、遮挡多，模型输出置信度偏低。降低到 0.10 可以减少漏检。实际使用时可对比不同 conf 下的 Precision-Recall 取舍。

---

## 十二、实验问题与改进方向

### 已知挑战

水下鱼群场景存在以下检测难点：

- **光照不均**：水下光照条件变化大，部分区域过暗或过曝
- **鱼体尺寸较小**：远距离鱼类在画面中像素占比小，小目标检测困难
- **鱼群遮挡严重**：密集鱼群中个体相互遮挡，造成漏检
- **背景干扰**：水草、礁石、气泡等易被误检为鱼
- **跨序列泛化**：不同视频序列的水质、视角、鱼种差异较大

因此 YOLOv8n（nano）在独立测试集上可能出现漏检和计数偏少的情况。

### 改进方向

- 使用更大模型：`yolov8s.pt` 或 `yolov8m.pt`
- 提高输入分辨率：`--imgsz 640` 或 `960`
- 增加训练轮数并配合更丰富的数据增强
- 扩充训练序列（纳入更多水下场景）
- 提升标注质量，减少漏标和错标
- 尝试更适合小目标/密集目标的检测架构

---

## 十三、GitHub 提交注意事项

以下文件和目录**不应提交到 Git 仓库**（已在 `.gitignore` 中配置）：

- `output/` — 所有输出文件
- `*.pt` — 模型权重文件（体积大）
- `*.mp4`、`*.avi`、`*.mov`、`*.mkv`、`*.wmv` — 视频文件
- `data/` — 原始数据集（体积大）
- `__pycache__/`、`*.pyc` — Python 缓存

如需调整 `.gitignore`，当前配置如下：

```gitignore
# pixi environments
.pixi/*
!.pixi/config.toml

# output files
output/

# model weights
*.pt

# video files
*.mp4
*.avi
*.mov
*.mkv
*.wmv

data/
```
