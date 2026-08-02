# 模型训练、导出与 KModel 转换

本目录提供钢球检测模型从 YOLOv8n 训练到 K230 部署模型的完整工具链，并包含本项目使用的数据集和可直接部署的 KModel。以下命令均在仓库根目录执行。

## 文件说明

| 文件 | 用途 |
| --- | --- |
| `yolo_train.py` | 使用 `yolov8n.pt` 训练单类别钢球检测模型 |
| `yolo_convert.py` | 将训练权重导出为固定输入尺寸的 ONNX |
| `export_yolo_roi_topk.py` | 在 ONNX 尾部加入 TopK16，并验证输出结果 |
| `to_kmodel.py` | 使用 nncase 将 TopK ONNX 转换为 KModel |
| `requirements-training.txt` | 训练与 ONNX 处理环境依赖 |
| `requirements-k230-convert.txt` | KModel 转换环境依赖 |
| `data_set/` | 本项目的训练集、验证集、标签与数据配置 |
| `yolov8n_roi_320_30_topk.kmodel` | 可直接部署到 K230 的 TopK16 模型 |

## 环境准备

建议分别创建训练环境和 KModel 转换环境，避免 ONNX 与 nncase 依赖冲突。

### 训练与 ONNX 环境

当前项目使用的主要版本：

- Python `3.11.15`
- Ultralytics `8.4.107`
- PyTorch `2.13.0+cu130`
- torchvision `0.28.0+cu130`
- ONNX `1.22.0`
- ONNX Runtime `1.28.0`

请先安装与本机 CUDA 环境匹配的 PyTorch，再安装其余依赖：

```powershell
conda activate yolo
python -m pip install -r .\training\requirements-training.txt
```

### KModel 转换环境

当前项目使用的主要版本：

- Python `3.11.15`
- nncase `2.11.0`
- ONNX `1.15.0`
- ONNX Runtime `1.19.0`
- onnxsim `0.4.36`

```powershell
conda activate k230_convert
python -m pip install -r .\training\requirements-k230-convert.txt
```

使用转换脚本前，需要按照 K230/nncase 环境要求完成相关插件配置。

## 数据集

仓库已在 `training/data_set/` 中包含本项目使用的钢球数据集和 `data.yaml`，可直接运行训练脚本。当前数据统计如下：

| 划分 | 图片 | 标签文件 | 目标框 | 图片尺寸分布 |
| --- | ---: | ---: | ---: | --- |
| 训练集 | 347 | 347 | 438 | 95 张 `320×30`；252 张 `640×60` |
| 验证集 | 57 | 57 | 87 | 29 张 `320×30`；28 张 `640×60` |

图像与标签按文件名一一对应，检测类别为 `0: ball`。如需增加或替换数据，请保持 `images/train`、`images/val`、`labels/train` 和 `labels/val` 的目录结构，并同步更新 `data.yaml`。详细格式见 [data_set/README.md](data_set/README.md)。

## 1. 训练 YOLOv8n

```powershell
conda activate yolo
python .\training\yolo_train.py
```

训练脚本使用以下关键参数：

| 参数 | 值 |
| --- | --- |
| 基础模型 | `yolov8n.pt` |
| 训练轮数 | `100` |
| 输入长边 | `320` |
| 矩形训练 | `rect=True` |
| Mosaic | `0.0` |
| Batch | `-1`，自动选择 |

`640×60` ROI 图像等比例缩放为 `320×30` 后，上下各填充 1 行，实际训练张量为 `[B, 3, 32, 320]`。

## 2. 导出 ONNX

将最佳权重复制到导出脚本约定的位置。若 Ultralytics 生成了 `train2` 等目录，请按实际训练目录调整源路径。

```powershell
Copy-Item .\runs\detect\train\weights\best.pt .\training\yolov8n_roi_320_30.pt
python .\training\yolo_convert.py
```

导出的模型契约为：

```text
input  images  [1, 3, 32, 320]
output output0 [1, 5, 210]
```

## 3. 添加 TopK16

```powershell
python .\training\export_yolo_roi_topk.py
```

脚本生成 `training/yolov8n_roi_320_30_topk.onnx`，将输出压缩为 `[1, 5, 16]`。生成后会使用固定随机输入，对比 TopK ONNX 与 NumPy 排序结果。

TopK16 只减少输出候选框数量；置信度过滤、坐标回映和 NMS 仍在 K230 端执行。

## 4. 转换 KModel

仓库已提供 `yolov8n_roi_320_30_topk.kmodel`。以下命令用于从 TopK ONNX 重新转换 KModel：

```powershell
conda activate k230_convert
python .\training\to_kmodel.py `
  --target k230 `
  --model .\training\yolov8n_roi_320_30_topk.onnx `
  --dataset .\training\data_set\images\train `
  --input_width 320 `
  --input_height 32 `
  --ptq_option 0
```

当前转换脚本按文件名排序，从校准目录中选择前 5 张图片。请确保目录中至少包含 5 张具有代表性的钢球 ROI 图像。

生成文件：

```text
training/yolov8n_roi_320_30_topk.kmodel
```

## 5. 部署到 K230

将仓库中的 `training/yolov8n_roi_320_30_topk.kmodel` 复制到 `/data/yolov8n_roi_320_30_topk.kmodel`，并按照 [K230 板端部署说明](../k230/README.md)复制运行代码。