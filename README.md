# Endfield-Map-CLS

基于 YOLOv26 分类模型的明日方舟·终末地小地图区域识别系统。  
从游戏截图中裁取小地图区域，通过分类器判断当前所处的地图分区，供自动化框架（如 MaaFramework）进行地图导航决策。

---

## 工作原理

推理分为两个阶段，分别由不同层负责：

```
【调用层 — MapLocator】
游戏截图 (1280×720)
    │
    ▼  按分辨率比例缩放至 720p 基准
    ▼  按固定坐标裁取小地图 ROI (x=49, y=51, w=118, h=120)
    │
    ▼
小地图 Mat (约 118×120, BGR)
    │
    ▼
【推理层 — YoloPredictor::predictZoneByYOLO】
    ▼  居中放置于 128×128 黑色画布
    ▼  应用半径 53px 圆形 Mask（消除小地图外框噪声）
    ▼  BGR→RGB + 归一化 [0, 1] + 转 NCHW tensor
    │
    ▼
ONNX 分类器推理
    │
    ▼
输出类别名称（经 region_mapping 转换为 ZoneId）
```

`YoloPredictor` 接收的输入是**已裁取的小地图 Mat**，而非完整游戏截图。ROI 裁取由上层 `MapLocator` 完成后传入。

模型的训练样本由大地图切片生成：将全局地图按 0.16× 缩放后，以滑窗方式裁取局部区域，叠加旋转、光度畸变、遮挡等增强，确保训练分布与推理端实际接收的小地图 Mat 高度匹配。

---

## 环境准备

**Python 版本要求：** 3.10+

```bash
pip install ultralytics opencv-python numpy
```

> 如需 GPU 加速训练，请额外安装对应 CUDA 版本的 PyTorch，
> 并将 `onnxruntime-gpu` 替换为对应版本。完整依赖见 `requirements.txt`。

---

## 完整训练流程

### 第一步：准备原始地图素材

在项目根目录下创建 `source_images/` 目录，将各地图分区的**完整大地图截图**放入其中。

**命名规则：**
- 文件名（不含扩展名）即为类别名，例如 `Map01Base.png` → 类别 `Map01Base`
- 支持 `.png` / `.jpg`（大小写不敏感）
- 也可以用子目录组织：`source_images/Map01Base/`，此时**目录名**为类别名，目录内可放多张该区域的地图切片
- `None/` 是保留的特殊类别名，用于存放"当前未处于任何已知地图"的负样本截图（如加载界面、对话框、过场动画等），在该子目录下放图片即可

实际 `source_images/` 结构示例：

```
source_images/
├── Map01Base.png           <- 大世界一区基础层完整地图
├── Map01Lv001Tier114.png   <- 大世界一区 Lv1 某小区域切片
├── Map01Lv002Tier120.png
├── Map02Base.png           <- 大世界二区基础层完整地图
├── Dung01Base.png          <- 谷地像差地图
├── OMVBase01.png
├── None/                   <- 负样本：非地图界面截图
│   ├── 加载中.png
│   └── ...
└── ...
```

**关键约束：所有原始地图图片必须缩放至原始游戏地图的 0.16 倍**

```bash
# 示例：若游戏地图原图为 10000×8000，缩放后应为约 1600×1280
# 可使用任意图像处理工具完成缩放
```

这个缩放比例是与游戏小地图的裁取比例对齐的，不匹配则训练样本与推理输入的视野范围不一致，导致识别失败。

---

### 第二步：生成训练数据集

```bash
python preprocess.py --input source_images --output dataset
```

脚本会自动完成以下工作：
- 以步长 40px 的滑窗扫描每张地图，筛选纹理丰富的区域（过滤空旷/单调区块）
- 对每个有效区域叠加随机旋转（0°~360°）、光度畸变、中心 UI 仿真、随机遮挡等增强
- 随机合成背景（真实背景图、纯黑、纯白、彩色噪点）以增强泛化能力
- 每个类别生成约 3000 张样本，按 8:2 比例划分 `train/` 和 `val/`
- 全程多进程并行，CPU 核心越多越快

**可选参数：**

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--input` | `source_images` | 原始地图切片目录 |
| `--output` | `dataset` | 数据集输出目录 |
| `--error` | `error_images` | 困难负样本目录（见第三步） |
| `--bg` | `bg_images` | 背景域随机化图片目录 |

生成完成后，`dataset/` 目录结构如下：

```
dataset/
├── train/
│   ├── Map01Base/
│   │   ├── Map01Base_00000.jpg
│   │   └── ...
│   ├── Map01Lv001Tier114/
│   ├── Map02Base/
│   ├── None/
│   └── ...
└── val/
    ├── Map01Base/
    ├── Map01Lv001Tier114/
    └── ...
```

---

### 第三步（可选）：添加困难负样本

如果在实际运行中发现模型对某些区域识别错误，将这些游戏全屏截图直接通过 `preprocess_roi.py` 裁切处理，输出到 `error_images/` 对应目录，作为困难负样本加入训练。

**初始化困难负样本目录（一次性操作）：**

```bash
python init_error_dirs.py --source source_images --error error_images
```

这会在 `error_images/` 下自动创建与类别同名的子目录：

```
error_images/
├── Map01Base/
├── Map01Lv001Tier114/
├── Map02Base/
├── None/
└── ...
```

**收集识别失败样本：**

直接将原始全屏截图通过 `preprocess_roi.py` 裁切并输出到对应目录，无需手动裁切：

```bash
# 处理单张截图，输出到对应类别目录
python preprocess_roi.py -i screenshot.jpg -o error_images/Map01Base

# 批量处理一个目录下的所有截图
python preprocess_roi.py -i raw_failures/Map01Base -o error_images/Map01Base
```

脚本会自动从全屏截图中裁取小地图 ROI 并归一化为 128×128，直接落盘到 `error_images/<class_name>/`。

**重新生成数据集并训练：**

```bash
python preprocess.py --input source_images --output dataset --error error_images
```

脚本会对 `error_images/` 中的困难样本过采样 15 倍并并入训练集，然后正常执行 `train.py` 即可。

**准备背景图：**

将**纹理复杂、色彩丰富、与地图内容完全无关**的图片放入 `bg_images/`，预处理时会随机裁取其中的区块作为合成背景。

背景图的质量直接影响模型的抗遮挡能力——背景越复杂、色彩越多样，模型在被各种 UI 元素、特效遮挡时的鲁棒性越强。纯色图、简单渐变、低饱和度的截图效果较差，应避免使用。

---

### 第四步：训练模型

```bash
python train.py
```

**首次训练**会自动从 `yolo26s-cls.pt` 底模开始。  
**再次训练**（增量微调）会自动检测 `runs/classify/` 下最新的 `best.pt` 并以此为起点继续训练。

**常用参数：**

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--data` | `dataset` | 数据集根目录 |
| `--model` | `auto` | 权重路径，`auto` 自动发现最新历史权重 |
| `--epochs` | `200` | 最大训练轮数 |
| `--batch` | `128` | 批次大小（显存不足时调小） |
| `--patience` | `20` | 早停轮数（验证集无提升时停止） |
| `--device` | `0` | CUDA 设备编号，`cpu` 使用 CPU 训练 |
| `--name` | `train` | 本次实验名称（区分多次训练结果） |

训练结果保存于 `runs/classify/<name>/weights/best.pt`。

---

### 第五步：验证推理效果

`predict.py` 接收完整的游戏全屏截图，内部实现了与 C++ 两阶段等价的完整预处理流程（缩放 → 裁取 ROI → 居中/Mask/归一化），可用于在不依赖 C++ 环境的情况下快速验证模型效果。

```bash
python predict.py path/to/screenshot.jpg
```

输出示例：

```
>>> Predictions:
  Map01Lv001Tier114: 98.32%
  Map01Lv001Tier115:  1.21%
  ...
```

**调试模式**（检查预处理结果是否正确）：

```bash
python predict.py path/to/screenshot.jpg --debug
# 额外保存 debug_inference.jpg，可目视确认裁取的小地图区域与圆形 Mask 是否正确
```

---

### 第六步：导出 ONNX 供 C++ 推理端使用

```bash
python export.py
```

默认自动选取最新的 `best.pt` 并导出为 ONNX（opset 21），同时生成同名 `.json` 部署配置文件：

```json
{
    "input_name": "images",
    "output_name": "output0",
    "classes": ["Map01Base", "Map01Lv001Tier114", "Map02Base", "None", "..."],
    "region_mapping": {}
}
```

**`region_mapping`（可选）：** 可在 `deploy_meta.json` 中配置类别名到业务区域名的映射，导出时会自动合并：

```json
{
    "input_name": "images",
    "output_name": "output0",
    "region_mapping": {
        "Map01": "ValleyIV",
        "Map02": "Wuling"
    }
}
```

```bash
python export.py --meta deploy_meta.json
```

**导出参数：**

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--model` | 自动发现 | 指定 .pt 权重路径 |
| `--imgsz` | `128` | 导出模型的推理图像尺寸 |
| `--meta` | `deploy_meta.json` | 外部元数据配置文件路径 |

---

## 项目文件说明

```
Endfield-Map-CLS/
├── preprocess.py         # 数据集生成流水线，处理游戏解包的原始地图素材
├── preprocess_roi.py     # 困难负样本预处理工具，将原始全屏失败截图裁切为 128×128 训练格式
├── train.py              # 训练脚本（支持 auto 增量微调）
├── predict.py            # 单图推理脚本
├── export.py             # ONNX 导出 + 部署配置生成
├── init_error_dirs.py    # 困难负样本目录初始化工具
├── deploy_meta.json      # 部署元数据（region_mapping 等，按需修改）
├── source_images/        # 原始地图切片（用户自备）
├── error_images/         # 困难负样本（可选）
├── bg_images/            # 背景域随机化图片（可选）
├── dataset/              # 生成的训练数据集（preprocess.py 输出）
└── runs/                 # 训练结果（train.py 输出）
```

---

## 关键参数速查

| 参数 | 数值 | 说明 |
|---|---|---|
| 地图缩放比例 | **0.16×** | source_images 中的图片必须预先缩放 |
| 推理基准分辨率 | **1280×720** | ROI 坐标的标定分辨率；非 720p 截图需预先等比缩放至此基准再裁取 ROI |
| 模型输入尺寸 | **128×128** | 固定，不可更改 |
| 小地图 ROI | x=49, y=51, w=118, h=120 | 在 720p 截图中的坐标 |
| 圆形 Mask 直径 | **106px** | 有效区域直径 |
