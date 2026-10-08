# Endfield-Map-CLS

《明日方舟：终末地》小地图区域分类模型。从游戏截图裁出左上角小地图，用 YOLO26 分类模型判断当前所在的地图分区（Base 大图按 160px 格切成类别，Tier 层与无小地图画面各自成类），供 [MaaEnd](https://github.com/MaaEnd/MaaEnd) 等基于 [MaaFramework](https://github.com/MaaXYZ/MaaFramework) 的自动化框架做定位与寻路。

## 目录

- [核心规格](#核心规格)
- [环境](#环境)
- [工程流水线](#工程流水线)
  - [1. 原始数据池](#1-原始数据池)
  - [2. 数据集合成与增强](#2-数据集合成与增强)
  - [3. 困难样本与固定验证集](#3-困难样本与固定验证集)
  - [4. 模型训练](#4-模型训练)
  - [5. 推理验证、导出与更新报告](#5-推理验证导出与更新报告)
- [目录结构](#目录结构)

## 核心规格

| 规格项 | 约束值 | 说明 |
| :--- | :--- | :--- |
| 基础分辨率 | 1280×720 | ROI 坐标的基准，其他分辨率的输入先缩放到 720p |
| 小地图 ROI | x=49, y=51, w=118, h=120 | 720p 基准图上的裁剪坐标 |
| 遮罩 | 直径 106px 圆形 | 去掉小地图外框与圈外 UI |
| 模型输入 | 128×128 | 固定网络输入，不可修改 |
| 大地图缩放率 | 0.16× | 制作 `source_images` 时，游戏解包大图的缩放倍率 |

## 环境

依赖由 `pyproject.toml` 管理（Python ≥ 3.14），用 [uv](https://docs.astral.sh/uv/) 安装：

```bash
uv sync
uv run python preprocess.py   # 下文的 python 命令都可以这样在 uv 环境里执行
```

训练机需要带 CUDA 的 PyTorch：在该机器的 `pyproject.toml` 里加 PyTorch 的 CUDA 索引后再 `uv sync`，写法见 [uv 的 PyTorch 指南](https://docs.astral.sh/uv/guides/integration/pytorch/)。

## 工程流水线

### 1. 原始数据池

基础地图切片放在 `source_images/`：

1. 所有原始大图按 **0.16×** 预缩放，对齐游戏内小地图的视野。
2. 文件名（不含扩展名）或子目录名就是该区域的类别标签。
3. `None` 是保留类别，存放游戏处于非地图界面（加载、UI 面板等）的负样本。

```text
source_images/
├── Map01Base.png           # 大世界一区基础层
├── Map01Lv001Tier114.png   # 导出的 Tier 模板（与 map_export.json 一一对应）
└── None/                   # UI 负样本
    └── loading_screen.png
map_export.json             # 完整导出契约：Tier 与所属 Base 的仿射关系
```

看不到小地图的大世界画面放在 `scene_images/` 的分组子目录里：`world/` 放大世界整屏截图或切掉 HUD 的裁块，`zipline/` 放滑索滑行时推理区附近的裁块（滑行中小地图整个隐藏）。帧已是 720p 尺度、原样使用不缩放，两边都不小于 182px；根目录散放或出现别的子目录会直接报错。只收无损原图，录屏帧太糊不用。

### 2. 数据集合成与增强

```bash
python preprocess.py --input source_images --output dataset
```

含 Tier 模板时，根目录的 `map_export.json` 是必需输入。它是导出工具生成的独立 `map-cls-export-v2` JSON 契约；CLS 只读取其中的文件名、尺寸、`tier_to_parent` 仿射和合成前的原始前景掩码，不导入 Endfield-tools 或 MapTracker。缺少、过期或与 `source_images` 不一致时预处理直接失败，避免悄悄生成没有父图上下文的 Tier 样本。也可以显式指定路径：

```bash
python preprocess.py --map-export map_export.json
```

预处理按 CPU 核数开进程（`--workers` 可改）。Map01Base、Map02Base 按 160px 格拆成独立任务，None 小任务排在队尾，进程池全程满载；日志里的 `[完成数/总数]` 是任务进度。命令行覆盖的配置（如 `--target-count`）会传给每个子进程。

> 一次生成几十万张小图。杀毒软件的实时文件监控（如火绒）会把写盘拖到每个进程每秒几张、CPU 只剩 15–20%，跑之前把项目目录加进信任区。

#### 采样与配额

- **均衡滑窗**：步长 8px，自动剔除低信息熵区块，并按完整轮次均衡覆盖所有有效中心。
- **空间扰动**：重复轮次使用不跨 tile 边界的 ±5px 位移和 ±0.5° 旋转；小地图尺度固定（实机相对源图 1.00–1.06），预处理不做缩放。
- **Tier 合法中心**：导出时在填入 Base 前提取原始 Tier 前景，以行优先 `[起点, 长度]` 写入 JSON 的 `foreground_runs`；普通、圈、中心图标及位移全部受此掩码约束，不从合成后的亮度反推。Base 另外排除指针落在纯黑空洞的位置，保留灰色地图阴影。
- **任务圈与配额**：普通类别仍为 1200 张基础样本加 600 张圈样本。Tier 每类总量为合法中心数 × 6，限制在 510–1000，再按八月 `e57968a` 的普通 `300`、区域圈 `150`、中心图标 `max(合法中心数, 60)` 相对权重分桶，并保证中心图标覆盖全部合法中心；不再使用 6:2:2。合法中心锁在高亮前景后数量约减半，固定 1000 张会让每个中心的训练样本升到八月的 4 倍，相邻 Tier 与 Base 实机帧的概率被重复中心吸走。黄/蓝圈与密集/超密集图标保留在各桶内，不重复追加数量。Base 的密集/超密集图标配额按整类 1200 张主样本计算，只落在可增强样本上，锚点不占配额。
- **集划分**：可切分样本按 8:2 划分 train/val，训练专用增强不参与验证集数量计算，人工困难样本只进入 train。

#### 地图层

- **地图层模糊**：30% 增强样本在叠加 UI 前对地图层做 σ0.4–1.0 高斯模糊，图标保持清晰，覆盖实机小地图的插值平滑。
- **地图层斑驳**：50% 增强样本在叠加 UI 前给地图层乘上 1–2px 块状亮度噪声（σ0.04–0.15），图标保持干净。实机小地图对高分辨率贴图点采样，地面带亮度比例的斑驳，导出底图是平滑的；在合成帧上加这种斑驳能把正确类得分从 99.8% 打到 19.0%，与 issue 5571 实机帧一致。
- **地图层色调**：50% 增强样本在叠加 UI 前把地图层对比度向中灰支点收拢（增益 0.60–0.95，支点亮度 70–130），图标保持纯白。取值来自 14 张实机帧对导出底图的逐像素拟合（增益 0.63–0.93、偏置 +5–+45）；训练期的整图 HSV 扰动会连图标一起变暗，覆盖不到这种差异。
- **视野扇形**：85% 增强样本在叠加 UI 前画相机视野扇形：从玩家点发出的白色半透明扇面，方向随机（与指针朝向无关），半角 25–36°，顶点 α 0.35–0.60，沿半径线性淡出到 30–48px，图标与指针盖在它上面。实机每一帧都有这块扇面，取值来自 7 张已定位实机帧的拟合；训练集缺它时模型对它的反应忽正忽负（同一帧抹掉扇形，同一次训练的两个权重一个 +11pp、一个 −22pp），固定验证帧逐轮摆动。
- **圆边透出**：85% 增强样本（Base 黑底干净锚点与 None 除外）在合成背景时把地图层外圈按半径渐隐，透出与透明区同一张背景：透出比例 = 峰值 0.80–1.00 × ((r − 起点) / 17.5)²，起点 38–42px，图标、扇形与指针画在上面保持原样。取值来自 21 张已定位 Win32 实机帧的逐半径拟合（r44 约 7%、r49 23%、r53 遮罩边 51%，透出场景约为圈外亮度的 0.8 倍）；训练集缺这一圈时，近黑地面格叠上合成圆边，正确类得分从 85% 掉到 49%，应龙关 r05_c06 实机帧因此被认成 r03_c09。

#### UI 叠加

- **玩家指针**：所有地图正样本始终叠加中心玩家指针，普通图标与路线独立随机出现。
- **环境仿真**：光度畸变、UI 遮挡仿真、中心角色标记位仿真。
- **滑索指示**：UI 线条之外按 70% 概率叠加滑索方向箭头链：青白 V 形箭头约 5px 一个，箭头尖朝折线前进方向，外带淡青柔光。取点方式与黄/白路线相同，在小地图内随机贯穿，方向随机。
- **中心设施簇**：密集配额之外再从普通样本划出约 3.4%，在玩家指针 ±8px 内叠加 12–32 个、仅 2–4 种的设施图标，占圆内 10–38%（均值约 21%），60% 的簇在图标下方再画 2–5 条暗黄半透明电线，模拟玩家在淤积点、矿点周围扎堆建造并拉供电网（issue 5833/5809/5571）。Tier 中心图标簇有 35% 改用这种设施簇，补足洞穴类 Tier 的高密度样本。类别总量、密集与超密集数量保持原样；这类样本与其他密集样本一样只进 train。

#### 背景

- **背景域泛化**：Base 每个合法中心只保留一次黑底干净锚点，其余出现全部进入图标与模糊增强，背景轮换纹理底与结构化场景（含跨地图干扰碎片）；Tier 恢复八月的黑底、纹理底、结构底 1:1:1，每个分桶都覆盖三种背景，不回滚成全黑底。纹理底与结构底各有一半取暗场景：亮度均值 12–60、对比度 0.15–0.60，对应实机透明区透出的被小地图底板压暗的场景（实测均值 27–38、标准差 8–9）；其余一半保持 4–244 的全亮度范围。Tier 样本四周恒为暗父图，Base 的暗背景过少时"暗色有纹理的四周"会变成 Tier 线索。
- **大世界场景**：`scene_images/` 按 `NONE_SCENE_TOTAL_CAP` 单独生成 None 样本，不占 UI None 的 3000 配额，先按 `SCENE_GROUP_SHARES`（world 0.75、zipline 0.25）切到各组，组内按帧均分，None 采样圆只落在源图内部；同时替换一半纹理/结构背景的程序化底，透出底按同一份额抽组，让地图类透明区透出同样分布的真实场景，模型判 None 只能看有没有地图层。

### 3. 困难样本与固定验证集

线上发现的误判截图可以直接沉淀为训练样本：

```bash
# 首次使用时初始化目录结构
python init_error_dirs.py --source source_images --error error_images

# 把现场回传的全屏错误截图切进对应类别（支持单图或整个目录）
python preprocess_roi.py -i raw_failures/Map01Base -o error_images/Map01Base

# 带上困难样本重建数据集
python preprocess.py --input source_images --output dataset --error error_images
```

同一类别的困难样本只加载一次，默认每张至少重复 5 次；现场样本很少时补足到该类生成样本量的 5%，确保它们有实际训练权重，同时不进入随机验证集造成数据泄漏。

固定真实验证样本放在 `validation_images/<class_name>/`，必须是已经按线上推理规格处理好的 128×128 图片，地图类与 `None` 都可以放。`preprocess.py` 会校验标签与尺寸并自动复制到 `dataset/val`；训练时它们的平均损失和最差样本损失会与生成验证集共同决定 `best.pt` 和早停。详细约束见 [`validation_images/README.md`](validation_images/README.md)。

### 4. 模型训练

基于 Ultralytics 训练：

```bash
python train.py --epochs 200 --batch 128 --device 0
```

- **训练期增强**：每轮以玩家像素为基准随机放大（面积 U(0.7,1)，倍率 1.00–1.195，双线性）、±6px 平移、50% 水平镜像，随后按 106px 圆形重新遮罩；其中 40% 的样本走不放大、整数像素平移的支路，像素不经插值原样进网络，保住地图层斑驳的实机强度。平移小于滑窗步长 8px，Base 中心留在本 tile；镜像恢复八月配方，稳的模型都是翻转不变的。长宽比拉伸关闭；保留颜色扰动。
- **CUDA Graphs**：默认开启 `torch.compile` 的 `reduce-overhead`。128px 小模型每步瓶颈在 Python 下发 GPU 算子，开启后 batch 与迭代次数不变，单步耗时约降 40%；启动时编译约 1 分钟，训练集丢弃每轮不足一个 batch 的尾部样本。排查问题或机器上没有 triton 时用 `--compile False` 回到 eager。
- **断点续训**：训练中断后用 `--resume runs/classify/.../weights/last.pt` 接着跑，轮数、优化器、EMA、学习率进度与输出目录都取自检查点，batch、workers、device、patience 按本次参数；中途不要重跑 `preprocess.py`。
- **起点权重**：首次训练从 `yolo26s-cls.pt` 开始；默认的 `--model auto` 会拿 `runs/classify/` 下最新的 `best.pt` 开一轮新的微调，学习率从头走，接着中断的训练要用 `--resume`。
- **无人值守**：云 GPU 上加 `--shutdown`，早停、跑满轮数或异常退出后自动关机，避免空转计费；权重与 `results.csv` 每轮已落盘。

> Windows 上 DataLoader 报 `Couldn't open shared file mapping ... error code: <1455>` 是页面文件太小、提交内存到顶：把页面文件设到空间充足的盘（如 32–64 GB）后重启。系统盘快满时，系统自动管理的页面文件长不大。

### 5. 推理验证、导出与更新报告

单张截图验证（内置与 C++ 端一致的预处理）：

```bash
python predict.py path/to/screenshot.jpg --debug
# --debug 会落盘 debug_inference.jpg，用于核对 ROI 与遮罩
```

导出 ONNX：

```bash
python export.py --imgsz 128 --meta deploy_meta.json
```

默认取最新权重（`--model` 可指定），输出 `best.onnx`（opset 21）及部署描述文件 `best.json`。

<details>
<summary>deploy_meta.json 配置示例</summary>

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

发版时生成模型更新报告图贴进 PR：

```bash
python report.py --online 23ef080 --candidate c674cef --online-name best35 --candidate-name best42
```

`--online` / `--candidate` 接 MaaEnd-AI 的提交号（取该提交的 `map/cls.onnx` 与 `cls.json`），也可以直接给导出的 `best.onnx`。报告用 `validation_images` 的全部固定真实验证帧对比两版：认对数、正确类平均与最低置信度、低于 Aug11（MaaEnd-AI `3a822e8`）超 1 个百分点的样本数，逐样本列出置信度有变化的帧，并附变化最大的几张原图；格号按 MaaEnd 的地图布局换成区域名。输出 `model_report.png`。

## 目录结构

```text
Endfield-Map-CLS/
├── preprocess.py               # 数据管线：素材切片与增强合成
├── preprocess_roi.py           # 全屏截图 → 小地图 ROI（困难样本归档）
├── preprocess_roi_auto_res.py  # 同上，按 720p 基准自动缩放的批处理版
├── init_error_dirs.py          # 按 source_images 初始化 error_images 类别目录
├── assign_error_tiles.py       # Base 错图按模型推理自动分配到格类别
├── review_tile_tool.py         # 在 Base 原图上画格网，核对 tile 映射
├── train.py                    # 训练
├── fine_tune.py                # 从最新候选权重低学习率微调
├── predict.py                  # CLI 推理验证
├── export.py                   # 导出 ONNX 与部署描述
├── report.py                   # 模型更新报告图（线上 vs 候选）
├── deploy_meta.json            # 部署元数据
├── map_export.json             # [输入] VFS 导出的 Tier→Base 契约
├── source_images/              # [输入] 基础素材
├── scene_images/               # [输入] 无小地图的大世界画面，分 world/ 与 zipline/
├── icon/                       # [输入] 小地图 UI 图标素材
├── error_images/               # [输入] 困难样本池
├── validation_images/          # [输入] 固定真实验证集
├── tests/                      # 单元测试
├── dataset/                    # [临时] 生成的训练集
└── runs/                       # [输出] 训练日志与权重
```
