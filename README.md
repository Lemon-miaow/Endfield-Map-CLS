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
独立 `map-cls-export-v2` JSON 契约；CLS 只读取其中的文件名、尺寸、`tier_to_parent` 仿射和合成前的原始前景掩码，不导入
Endfield-tools 或 MapTracker。缺少、过期或与 `source_images` 不一致时预处理会直接失败，
避免悄悄生成没有父图上下文的 Tier 样本。也可以显式指定路径：

```bash
python preprocess.py --map-export map_export.json
```

**管线处理细节：**

* **均衡滑窗**：步长 8px，自动剔除低信息熵区块，并按完整轮次均衡覆盖所有有效中心。
* **空间扰动**：重复轮次使用不跨 tile 边界的 ±5px 位移和 ±0.5° 旋转；小地图尺度固定（实机相对源图 1.00–1.06），预处理不做缩放。
* **地图层模糊**：30% 增强样本在叠加 UI 前对地图层做 σ0.4–1.0 高斯模糊，图标保持清晰，覆盖实机小地图的插值平滑。
* **地图层斑驳**：50% 增强样本在叠加 UI 前给地图层乘上 1–2px 块状亮度噪声（σ0.04–0.15），图标保持干净。实机小地图对高分辨率贴图点采样，地面带亮度比例的斑驳，导出底图是平滑的；在合成帧上加这种斑驳能把正确类得分从 99.8% 打到 19.0%，与 issue 5571 实机帧一致。
* **视野扇形**：85% 增强样本在叠加 UI 前画相机视野扇形：从玩家点发出的白色半透明扇面，方向随机（与指针朝向无关），半角 25–36°，顶点 α 0.35–0.60，沿半径线性淡出到 30–48px，图标与指针盖在它上面。实机每一帧都有这块扇面，取值来自 7 张已定位实机帧的拟合；训练集缺它时模型对它的反应忽正忽负（同一帧抹掉扇形，同一次训练的两个权重一个 +11pp、一个 −22pp），固定验证帧逐轮摆动。
* **地图层色调**：50% 增强样本在叠加 UI 前把地图层对比度向中灰支点收拢（增益 0.60–0.95，支点亮度 70–130），图标保持纯白。取值来自 14 张实机帧对导出底图的逐像素拟合（增益 0.63–0.93、偏置 +5–+45）；训练期的整图 HSV 扰动会连图标一起变暗，覆盖不到这种差异。
* **中心设施簇**：密集配额之外再从普通样本划出约 3.4%，在玩家指针 ±8px 内叠加 12–32 个、仅 2–4 种的设施图标，占圆内 10–38%（均值约 21%），60% 的簇在图标下方再画 2–5 条暗黄半透明电线，模拟玩家在淤积点、矿点周围扎堆建造并拉供电网（issue 5833/5809/5571）。Tier 中心图标簇有 35% 改用这种设施簇，补足洞穴类 Tier 的高密度样本。类别总量、密集与超密集数量保持原样；这类样本与其他密集样本一样只进 train。
* **Tier 合法中心**：导出时在填入 Base 前提取原始 Tier 前景，以行优先 `[起点, 长度]` 写入 JSON 的 `foreground_runs`；普通、圈、中心图标及位移全部受此掩码约束，不从合成后的亮度反推。Base 另外排除指针落在纯黑空洞的位置，保留灰色地图阴影。
* **训练加载**：每轮以玩家像素为基准随机放大（面积 U(0.7,1)，倍率 1.00–1.195，双线性）、±6px 平移、50% 水平镜像，随后按 106px 圆形重新遮罩；其中 40% 的样本走不放大、整数像素平移的支路，像素不经插值原样进网络，保住地图层斑驳的实机强度。平移小于滑窗步长 8px，Base 中心留在本 tile；镜像恢复八月配方，稳的模型都是翻转不变的。长宽比拉伸关闭；保留颜色扰动。
* **环境仿真**：光度畸变、UI 遮挡仿真、中心角色标记位仿真。  
* **玩家指针**：所有地图正样本始终叠加中心玩家指针，普通图标与路线独立随机出现。
* **滑索指示**：UI 线条之外按 70% 概率叠加滑索方向箭头链：青白 V 形箭头约 5px 一个，箭头尖朝折线前进方向，外带淡青柔光。取点方式与黄/白路线相同，在小地图内随机贯穿，方向随机。
* **任务圈增强**：普通类别仍为 1200 张基础样本加 600 张圈样本。Tier 每类总量为合法中心数 × 6，限制在 510–1000，再按八月 `e57968a` 的普通 `300`、区域圈 `150`、中心图标 `max(合法中心数, 60)` 相对权重分桶，并保证中心图标覆盖全部合法中心；不再使用 6:2:2。合法中心锁在高亮前景后数量约减半，固定 1000 张会让每个中心的训练样本升到八月的 4 倍，相邻 Tier 与 Base 实机帧的概率被重复中心吸走。黄/蓝圈与密集/超密集图标保留在各桶内，不重复追加数量。Base 的密集/超密集图标配额按整类 1200 张主样本计算，只落在可增强样本上，锚点不占配额。
* **背景域泛化**：Base 每个合法中心只保留一次黑底干净锚点，其余出现全部进入图标与模糊增强，背景轮换纹理底与结构化场景（含跨地图干扰碎片）；Tier 恢复八月的黑底、纹理底、结构底 1:1:1，每个分桶都覆盖三种背景，不回滚成全黑底。 纹理底与结构底各有一半取暗场景：亮度均值 12–60、对比度 0.15–0.60，对应实机透明区透出的被小地图底板压暗的场景（实测均值 27–38、标准差 8–9）；其余一半保持 4–244 的全亮度范围。Tier 样本四周恒为暗父图，Base 的暗背景过少时"暗色有纹理的四周"会变成 Tier 线索。
* **集划分**：可切分样本按 8:2 划分 train/val，训练专用增强不参与验证集数量计算，人工困难样本只进入 train。

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

固定真实验证样本放在 `validation_images/<class_name>/`，必须是已经按线上推理规格处理好的 128×128 图片。`preprocess.py` 会校验标签与尺寸并自动复制到 `dataset/val`；训练时它们的平均损失和最差样本损失会与生成验证集共同决定 `best.pt` 和 early-stop。详细约束见 [`validation_images/README.md`](validation_images/README.md)。

### **4.模型训练**

基于 Ultralytics 引擎执行训练，脚本支持智能断点续训及基座模型挂载。

```bash
python train.py --epochs 200 --batch 128 --device 0
```

云 GPU 无人值守时加 `--shutdown`：早停、跑满轮数或异常退出后自动关机，避免空转计费；权重与 `results.csv` 每轮已落盘。

训练默认开启 `torch.compile` 的 `reduce-overhead`（CUDA Graphs）：128px 小模型每步瓶颈在 Python 下发 GPU 算子，开启后 batch 与迭代次数不变，单步耗时约降 40%；启动时编译约 1 分钟，训练集丢弃每轮不足一个 batch 的尾部样本。排查问题时可用 `--compile False` 回到 eager。

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
│  
├── 📁 dataset/              # [Temp] 编译生成的训练集  
└── 📁 runs/                 # [Output] 训练日志与权重产物  
```
