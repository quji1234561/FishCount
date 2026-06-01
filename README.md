# 基于 YOLOv8 的水下鱼类目标检测与数量统计

## 一、项目简介

本项目是一个基于 **YOLOv8** 的水下鱼类目标检测与数量统计系统，作为《计算机视觉》课程设计项目。

项目通过解析 MOT 格式的 `gt.txt` 标注文件构建 YOLO 数据集，训练 YOLOv8 检测模型，实现对水下图片或视频中鱼类的自动检测与数量统计，并将结果可视化输出。

***

## 二、项目功能

- **数据格式转换**：将 MOT 格式的 `gt.txt` 标注文件转换为 YOLO 训练格式
- **数据集自动划分**：支持随机或顺序划分训练集/验证集
- **YOLOv8 模型训练**：基于 YOLOv8 预训练权重进行迁移学习
- **图片/视频鱼类检测**：对单张图片、图片文件夹或视频进行目标检测
- **鱼数量统计**：统计每张图片或每帧视频中的鱼类数量
- **检测结果可视化**：输出带检测框和数量标注的结果图片或视频
- **统计结果导出**：生成 `fish_count.csv` 汇总文件

***

## 三、项目结构

```
FishCount/
├── data/
│   └── BT-001/
│       ├── det/                      # 检测结果标注（可选）
│       │   ├── det.txt
│       │   ├── det_train_half.txt
│       │   └── det_val_half.txt
│       ├── gt/                       # 真值标注
│       │   ├── gt.txt
│       │   ├── gt_train_half.txt
│       │   └── gt_val_half.txt
│       └── img1/                     # 图片帧（000001.jpg ~ 003000.jpg）
├── yolo/
│   ├── gt_to_yolo.py                 # gt.txt → YOLO 数据集转换
│   ├── train_yolo.py                 # YOLOv8 模型训练
│   ├── predict_yolo.py               # YOLOv8 图片/视频预测
│   └── count_fish.py                 # 鱼数量统计
├── output/                           # 输出目录（自动生成）
│   ├── yolo_dataset/                 # 转换后的 YOLO 数据集
│   ├── yolo_runs/                    # 训练与预测结果
│   └── count_results/                # 鱼数量统计结果
├── pixi.toml                         # 项目依赖配置
├── .gitignore
└── README.md
```

***

## 四、环境依赖

本项目使用 **pixi**（基于 conda-forge）管理 Python 环境与依赖。

**Python 版本**：Python 3.13+

**主要依赖**：

| 依赖            | 版本             | 用途           |
| ------------- | -------------- | ------------ |
| `ultralytics` | ≥8.0.0         | YOLOv8 训练与推理 |
| `pytorch`     | ≥2.0.0, <2.11  | 深度学习框架       |
| `torchvision` | ≥0.17.0, <0.27 | 视觉工具集        |
| `pillow`      | ≥10.0.0        | 图像读取与处理      |
| `opencv`      | ≥4.13.0, <5    | 视频读写与绘图      |

**安装方式**：

```bash
# 安装 pixi（如未安装）
# Windows: winget install prefix-dev.pixi
# 或参考: https://pixi.sh/latest/#installation

# 安装项目依赖
pixi install
```

所有 Python 命令均需通过 `pixi run python` 执行，例如：

```bash
pixi run python yolo/gt_to_yolo.py ...
```

> 提示：在 VS Code 中将 Python 解释器切换为 `.pixi\envs\default\python.exe` 即可直接使用 IDE 运行与调试。

***

## 五、使用方法

以下命令均在项目根目录下执行，Windows CMD 使用 `^` 换行，Linux/macOS 使用 `\` 换行。

### 1. gt.txt 转 YOLO 数据集

将 MOT 格式标注转换为 YOLO 训练格式，自动划分训练集与验证集。

**Windows CMD**：

```batch
python yolo/gt_to_yolo.py ^
  --gt data/BT-001/gt/gt.txt ^
  --img-dir data/BT-001/img1 ^
  --output output/yolo_dataset ^
  --val-ratio 0.2 ^
  --seed 42
```

**Linux/macOS**：

```bash
python yolo/gt_to_yolo.py \
  --gt data/BT-001/gt/gt.txt \
  --img-dir data/BT-001/img1 \
  --output output/yolo_dataset \
  --val-ratio 0.2 \
  --seed 42
```

**参数说明**：

| 参数              | 默认值                   | 说明                                  |
| --------------- | --------------------- | ----------------------------------- |
| `--gt`          | 必填                    | gt.txt 文件路径                         |
| `--img-dir`     | 必填                    | 图片帧目录路径                             |
| `--output`      | `output/yolo_dataset` | 输出目录                                |
| `--val-ratio`   | `0.2`                 | 验证集比例                               |
| `--seed`        | `42`                  | 随机种子                                |
| `--class-name`  | `fish`                | 类别名称                                |
| `--split-mode`  | `random`              | 划分方式：`random`（随机）/ `sequential`（顺序） |
| `--filter-conf` | 关闭                    | 过滤 conf≤0 的标注                       |

输出目录结构：

```
output/yolo_dataset/
├── data.yaml                    # 数据集配置文件
├── images/
│   ├── train/                   # 训练图片（硬链接）
│   └── val/                     # 验证图片（硬链接）
└── labels/
    ├── train/                   # 训练标签（YOLO txt 格式）
    └── val/                     # 验证标签（YOLO txt 格式）
```

### 2. 训练 YOLOv8 模型

**Windows CMD**：

```batch
  --data output/yolo_dataset/data.yaml ^
  --model yolov8n.pt ^
  --epochs 50 ^
  --imgsz 640 ^
  --batch 8 ^
  --project output/yolo_runs ^
  --name fish_yolov8n
```

**Linux/macOS**：

```bash
python yolo/train_yolo.py \
  --data output/yolo_dataset/data.yaml \
  --model yolov8n.pt \
  --epochs 50 \
  --imgsz 640 \
  --batch 8 \
  --project output/yolo_runs \
  --name fish_yolov8n
```

**参数说明**：

| 参数           | 默认值                | 说明                                          |
| ------------ | ------------------ | ------------------------------------------- |
| `--data`     | 必填                 | data.yaml 路径                                |
| `--model`    | `yolov8n.pt`       | 预训练模型（yolov8n.pt / yolov8s.pt / yolov8m.pt） |
| `--epochs`   | `50`               | 训练轮数                                        |
| `--imgsz`    | `640`              | 输入图片尺寸                                      |
| `--batch`    | `8`                | 批次大小（显存不足时可减小）                              |
| `--workers`  | `4`                | 数据加载线程数                                     |
| `--device`   | 自动                 | 训练设备（`0` = GPU 0，`cpu` = CPU）               |
| `--project`  | `output/yolo_runs` | 输出根目录                                       |
| `--name`     | `train`            | 本次训练名称                                      |
| `--patience` | `20`               | 早停耐心值                                       |
| `--exist-ok` | 关闭                 | 允许覆盖同名输出目录                                  |

训练完成后，最佳模型路径为：

```
output/yolo_runs/fish_yolov8n/weights/best.pt
```

### 3. 图片或视频预测

使用训练好的模型对图片或视频进行检测，输出带检测框的结果。

**Windows CMD**（预测图片文件夹）：

```batch
python yolo/predict_yolo.py ^
  --weights output/yolo_runs/fish_yolov8n/weights/best.pt ^
  --source data/BT-001/img1 ^
  --project output/yolo_runs ^
  --name predict_fish ^
  --conf 0.25
```

**参数说明**：

| 参数            | 默认值                | 说明                 |
| ------------- | ------------------ | ------------------ |
| `--weights`   | 必填                 | 模型权重路径             |
| `--source`    | 必填                 | 输入源（图片/图片文件夹/视频路径） |
| `--imgsz`     | `640`              | 预测输入尺寸             |
| `--conf`      | `0.25`             | 置信度阈值              |
| `--iou`       | `0.45`             | NMS IOU 阈值         |
| `--device`    | 自动                 | 预测设备               |
| `--project`   | `output/yolo_runs` | 输出根目录              |
| `--name`      | `predict`          | 输出名称               |
| `--save-txt`  | 关闭                 | 保存 YOLO 格式 txt 结果  |
| `--save-conf` | 关闭                 | txt 中保存置信度         |

输出目录：`output/yolo_runs/predict_fish/`

### 4. 鱼数量统计

使用训练好的模型对图片或视频进行检测，统计每张图片/每帧中的鱼数量，输出带数量标注的结果和 CSV 文件。

**Windows CMD**（统计图片）：

```batch
python yolo/count_fish.py ^
  --weights output/yolo_runs/fish_yolov8n/weights/best.pt ^
  --source data/BT-001/img1 ^
  --output output/count_results ^
  --conf 0.25
```

**Windows CMD**（统计视频）：

```batch
python yolo/count_fish.py ^
  --weights output/yolo_runs/fish_yolov8n/weights/best.pt ^
  --source your_video.mp4 ^
  --output output/count_results ^
  --conf 0.25
```

**参数说明**：

| 参数           | 默认值                    | 说明                 |
| ------------ | ---------------------- | ------------------ |
| `--weights`  | 必填                     | 模型权重路径             |
| `--source`   | 必填                     | 输入源（图片/图片文件夹/视频路径） |
| `--output`   | `output/count_results` | 输出目录               |
| `--imgsz`    | `640`                  | 预测输入尺寸             |
| `--conf`     | `0.25`                 | 置信度阈值              |
| `--iou`      | `0.45`                 | NMS IOU 阈值         |
| `--device`   | 自动                     | 预测设备               |
| `--class-id` | `-1`（所有类）              | 目标类别 ID（0 = fish）  |

输出内容：

```
output/count_results/
├── fish_count.csv                # 鱼数量统计表格
└── images/                       # 带检测框和 Fish Count 的结果图片
    ├── 000001_count.jpg
    └── ...
```

对于视频输入，输出为带 Fish Count 覆盖的 `.mp4` 文件。

***

## 六、输出结果说明

| 路径                                              | 说明                                           |
| ----------------------------------------------- | -------------------------------------------- |
| `output/yolo_dataset/`                          | 转换后的 YOLO 格式数据集，包含 images、labels 和 data.yaml |
| `output/yolo_dataset/data.yaml`                 | 数据集配置文件，指定训练/验证集路径和类别名                       |
| `output/yolo_runs/fish_yolov8n/`                | 训练输出，含权重文件、训练日志和指标图表                         |
| `output/yolo_runs/fish_yolov8n/weights/best.pt` | 训练得到的最佳模型权重                                  |
| `output/yolo_runs/fish_yolov8n/weights/last.pt` | 训练最后一个 epoch 的模型权重                           |
| `output/yolo_runs/predict_fish/`                | 预测可视化结果（ultralytics 原生输出）                    |
| `output/count_results/fish_count.csv`           | 鱼数量统计 CSV 文件                                 |
| `output/count_results/images/`                  | 带检测框和 Fish Count 数量标注的结果图片                   |

***

## 七、CSV 字段说明

`fish_count.csv` 包含以下字段：

| 字段           | 类型    | 说明                  |
| ------------ | ----- | ------------------- |
| `frame_id`   | int   | 图片序号或视频帧序号，从 1 开始编号 |
| `file_name`  | str   | 图片文件名或视频文件名         |
| `fish_count` | int   | 当前帧中检测到的鱼数量         |
| `avg_conf`   | float | 当前帧所有检测框的平均置信度      |

***

## 八、课程设计说明

本项目适合作为 **计算机视觉** 课程设计，主要体现以下知识点：

- **目标检测任务**：基于 YOLOv8 实现单类别（鱼类）目标检测
- **数据格式转换**：理解 MOT 标注格式与 YOLO 标注格式的差异，实现格式间的规范化转换
- **深度学习模型训练**：掌握 YOLOv8 模型的训练流程、超参数调节和迁移学习
- **检测结果可视化**：使用 OpenCV 绘制检测框与统计信息
- **简单应用功能**：在目标检测基础上实现鱼类数量统计的实用功能

项目流程清晰、代码模块化，适合初学者理解目标检测的完整流水线。

***

## 九、实验问题与改进方向

### 已知挑战

水下鱼群场景存在以下检测难点：

- **光照不均**：水下光照条件变化大，部分区域过暗或过曝
- **鱼体较小**：远距离鱼类在画面中占比小，小目标检测困难
- **背景干扰**：水草、礁石、气泡等背景元素易造成误检
- **遮挡严重**：鱼群密集时相互遮挡，导致漏检

YOLOv8n（nano）作为轻量模型，在以上场景下可能出现漏检或误检，检测精度有限。

### 改进方向

- 使用更大规模的模型，如 `yolov8s.pt` 或 `yolov8m.pt`
- 提高输入尺寸，例如 `--imgsz 960` 以保留更多小目标细节
- 增加训练轮数，配合数据增强策略
- 扩充训练数据集，增加更多水下场景和鱼种
- 提升标注质量，减少标注噪声和漏标
- 使用更适合小目标检测的模型结构（如 YOLOv8 的高分辨率分支）

***

## 十、注意事项

1. **输出目录管理**：`output/` 目录由脚本自动生成，建议不要将 `output/`、`weights/`、大型视频文件提交到 Git 仓库
2. **.gitignore 配置**：建议在 `.gitignore` 中添加：
   ```
   output/
   *.pt
   *.mp4
   *.avi
   ```
3. **数据路径检查**：训练前确保 `data.yaml` 中的路径正确指向实际图片目录
4. **显存不足**：如果 GPU 显存不足，可以减小 `--batch` 参数（例如 `--batch 4` 或 `--batch 2`）
5. **检测结果过少**：如果检测框太少，可以适当降低置信度阈值，例如 `--conf 0.15` 或 `--conf 0.1`
6. **路径规范**：所有脚本支持相对路径和绝对路径，建议在项目根目录下执行以避免路径错误
7. **pixi 环境**：如果 IDE 提示无法解析导入，请将 Python 解释器切换为 `.pixi\envs\default\python.exe`

