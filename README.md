# **Endfield-Map-CLS**

**Endfield-Map-CLS** 是一个专为《明日方舟：终末地》设计的轻量级小地图区域识别系统。本项目基于先进的 **YOLOv26 分类架构**，通过从游戏截图中提取并分析小地图区域，精准判断当前所处的地图分区。该组件可作为底层视觉感知模块，无缝集成至自动化框架（如 [MaaFramework](https://github.com/MaaAssistantArknights/MaaFramework)）中，为复杂的地图导航和寻路决策提供状态支撑。

## **📑 目录**

* [核心规格参数](#️-核心规格参数)  
* [环境依赖](#️-环境依赖)  
* [工程流水线 (Pipeline)](#-工程流水线-pipeline)  
  * [阶段一：原始数据池构建](#1原始数据池构建)  
  * [阶段二：数据集合成与增强](#2数据集合成与增强)  
  * [阶段三：困难样本挖掘 (Active Learning)](#3困难样本挖掘-active-learning)  
  * [阶段四：模型训练](#4模型训练)  
  * [阶段五：推理验证与导出](#5推理验证与导出)  
* [目录结构](#️-目录结构)

## **⚙️ 核心规格参数**

| 规格项 | 约束值 | 备注说明 |
| :---- | :---- | :---- |
| **基础分辨率** | 1280×720 | ROI 坐标标定的绝对基准，非该分辨率输入需预先缩放 |
| **小地图 ROI** | x=49, y=51, w=118, h=120 | 基于 720p 基准图的裁剪坐标 |
| **Mask 规格** | Diameter = 106px | 用于消除小地图外框及 UI 噪声的圆形掩膜 |
| **模型输入尺寸** | 128×128 | 固定网络输入，严禁修改 |
| **大地图缩放率** | 0.16× | 制作 source_images 时，游戏解包大图必须缩放的倍率 |

## **🛠️ 环境依赖**

推荐使用 Python 3.10 或更高版本。

```bash
# 基础依赖  
pip install ultralytics opencv-python numpy

# GPU 加速支持 (按需安装对应 CUDA 版本的 PyTorch)  
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## **🚀 工程流水线 (Pipeline)**

### **1.原始数据池构建**

在项目根目录下构建 `source_images/` 目录，用于存放预处理后的基础地图切片。

**📌 规范与约束：**

1. 所有原始大图必须严格按照 **0.16×** 的比例进行预缩放，以对齐游戏内小地图的真实 FOV。  
2. 命名规范：文件名（不含扩展名）或子目录名将作为该区域的**类别标签**。  
3. `None` 为系统保留类别，用于存放游戏处于非地图界面（如加载、UI面板等）的负样本。

```text
source_images/  
├── Map01Base.png           # 大世界一区基础层  
├── Map01Lv001Tier114.png   # 导出的 Tier 模板（与 map_export.json 一一对应）
└── None/                   # 纯净负样本池  
    └── loading_screen.png
map_export.json              # 完整导出契约：Tier 与所属 Base 的仿射关系
```

### **2.数据集合成与增强**

通过滑窗采样与数据增强管线，自动生成高泛化能力的训练数据集。

```bash
python preprocess.py --input source_images --output dataset
```

含 Tier 模板时，根目录的 `map_export.json` 是必需输入。它是导出工具生成的
独立 JSON 契约；CLS 只读取其中的文件名、尺寸和 `tier_to_parent` 仿射，不导入
Endfield-tools 或 MapTracker。缺少、过期或与 `source_images` 不一致时预处理会直接失败，
避免悄悄生成没有父图上下文的 Tier 样本。也可以显式指定路径：

```bash
python preprocess.py --map-export map_export.json
```

**管线处理细节：**

* **均衡滑窗**：步长 8px，自动剔除低信息熵区块，并按完整轮次均衡覆盖所有有效中心。
* **空间扰动**：重复轮次使用不跨 tile 边界的 ±5px 位移和 ±0.5° 旋转。
* **环境仿真**：光度畸变、UI 遮挡仿真、中心角色标记位仿真。  
* **玩家指针**：所有地图正样本始终叠加中心玩家指针，普通图标与路线独立随机出现。
* **任务圈增强**：普通类别默认保留 1200 张基础样本并额外生成 600 张黄色/浅蓝色圈样本；在默认 Base tile 配置下，train 中每个有效中心至少保留一组同位置普通/黄色圈样本。
* **背景域泛化**：从 bg_images/ 中随机提取复杂纹理，替换地图边界之外的背景，大幅提升模型抗 UI 遮挡能力。  
* **集划分**：生成样本按 8:2 划分 train/val，人工困难样本只进入 train。

### **3.困难样本挖掘 (Active Learning)**

若在实际业务中发现 False Positive 样本，可通过辅助脚本快速将游戏原图沉淀为训练集，进行针对性微调。

```bash
# 1. 首次使用时初始化目录结构  
python init_error_dirs.py --source source_images --error error_images

# 2. 将现场回传的错误截图（全屏）直接切入对应真实分类池  
# 支持单图或整个目录的批量化 ROI 裁切  
python preprocess_roi.py -i raw_failures/Map01Base -o error_images/Map01Base

# 3. 触发携带困难样本的重构建
python preprocess.py --input source_images --output dataset --error error_images
```

同一类别的困难样本只加载一次，默认每张至少重复 5 次；当现场样本很少时，会补足到该类生成样本量的 5%，确保它们具有实际训练权重，同时避免进入随机验证集造成数据泄漏。

固定真实验证样本放在 `validation_images/<class_name>/`，必须是已经按线上推理规格处理好的 128×128 图片。`preprocess.py` 会校验标签与尺寸并自动复制到 `dataset/val`；训练时它们会独立计算损失，与生成验证集共同决定 `best.pt` 和 early-stop。详细约束见 [`validation_images/README.md`](validation_images/README.md)。

### **4.模型训练**

基于 Ultralytics 引擎执行训练，脚本支持智能断点续训及基座模型挂载。

```bash
python train.py --epochs 200 --batch 128 --device 0
```

**💡 提示：** 首次训练自动挂载 `yolo26s-cls.pt`。增量微调时，引擎会自动寻址 `runs/classify/` 下最新的 `best.pt` 作为起点。

### **5.推理验证与导出**

**快速效果验证**（内置完整的 C++ 等效预处理管线）：

```bash
python predict.py path/to/screenshot.jpg --debug  
# --debug 参数将落盘 debug_inference.jpg，供可视化核验 ROI 与 Mask 精度
```

**ONNX 工业级导出**：

```bash
python export.py --imgsz 128 --meta deploy_meta.json
```

自动抓取最新权重，并输出 `best.onnx` (opset 21) 及配套的部署描述文件 `best.json`。配置示例如下：

<details>
<summary>点击查看 deploy_meta.json 配置示例</summary>

```json
{  
    "input_name": "images",  
    "output_name": "output0",  
    "region_mapping": {  
        "Map01Base": "ValleyIV_Main",  
        "Map02Base": "Wuling_Main"  
    }  
}
```

</details>

## **🗂️ 目录结构**

```text
Endfield-Map-CLS/  
├── 📄 preprocess.py         # 数据管线：素材切片与增强合成  
├── 📄 preprocess_roi.py     # 数据管线：全图 ROI 自动裁切（用于困难样本归档）  
├── 📄 train.py              # 训练调度器  
├── 📄 predict.py            # CLI 推理验证工具  
├── 📄 export.py             # 模型编译与导出工具  
├── 📄 init_error_dirs.py    # 工程初始化工具  
├── ⚙️ deploy_meta.json      # 部署元数据配置文件  
├── 📄 map_export.json       # [Input] VFS 导出的 Tier→Base 契约
│  
├── 📁 source_images/        # [Input] 基础素材集 (需自行准备)  
├── 📁 error_images/         # [Input] 困难样本池 (Active Learning)  
├── 📁 validation_images/    # [Input] 可提交的固定真实验证集
├── 📁 bg_images/            # [Input] 背景域随机化素材池  
│  
├── 📁 dataset/              # [Temp] 编译生成的训练集  
└── 📁 runs/                 # [Output] 训练日志与权重产物  
```
