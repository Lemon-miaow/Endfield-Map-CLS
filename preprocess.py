"""
preprocess.py — 训练数据集生成流水线

从原始地图切片（source_images）中提取旋转增强样本，叠加背景合成、
光度畸变、中心 UI 仿真、随机遮罩等数据增强手段，并按 train/val 分层
落盘，用于 MobileNet 分类器的有监督训练。

目录约定:
    source_images/                  原始地图切片根目录
        <class_name>.png            直接置于根目录时，文件名即为类别名
        <class_name>/               子目录时，目录名即为类别名
            *.png / *.jpg

    error_images/<class_name>/      困难负样本目录（可选）；每张图片将被
                                    过采样 ERROR_OVERSAMPLE 倍后并入训练集

    bg_images/                      背景域随机化图片目录（可选）

    dataset/                        输出数据集根目录（运行前自动清空）
        train/<class_name>/*.jpg
        val/<class_name>/*.jpg

用法:
    python preprocess.py [--input <dir>] [--output <dir>]
                         [--error <dir>] [--bg <dir>]
"""

import argparse
import concurrent.futures
import logging
import math
import os
import random
import shutil
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# 全局配置
# ---------------------------------------------------------------------------

CONFIG = {
    "OUTPUT_SIZE": 128,       # 输出训练图像的边长（像素）
    "MASK_DIAMETER": 106,     # 小地图圆形有效区域的直径
    "TARGET_COUNT": 3000,     # 每个类别目标生成的样本总数
    "VAL_RATIO": 0.2,         # 验证集比例
    "STRIDE": 40,             # 滑窗扫描步长
    "STD_THRESHOLD": 5.0,     # 有效 Patch 的最低灰度标准差（过滤空白区域）
    "OCCLUSION_COUNT": 2,     # 随机遮挡块的最大数量
    "OCCLUSION_SIZE": 25,     # 随机遮挡块的最大边长（像素）
    "ERROR_OVERSAMPLE": 15,   # 困难负样本的过采样倍数
}

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 基础 I/O 工具
# ---------------------------------------------------------------------------


def safe_imread(path, flags=cv2.IMREAD_COLOR):
    """读取图片，兼容路径中包含非 ASCII 字符（如中文）的情况。"""
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), flags)


def safe_imwrite(path, img):
    """写入图片，兼容路径中包含非 ASCII 字符（如中文）的情况。"""
    ext = os.path.splitext(str(path))[1]
    is_success, buf = cv2.imencode(ext, img)
    if is_success:
        buf.tofile(str(path))
    return is_success


# ---------------------------------------------------------------------------
# 图像预处理
# ---------------------------------------------------------------------------


def get_safe_size() -> int:
    """计算旋转安全尺寸。

    旋转任意角度后不产生黑边所需的最小方形边长，即原正方形对角线长度的上取整。
    """
    return int(math.ceil(math.sqrt(2 * CONFIG["OUTPUT_SIZE"] ** 2)))


def load_image(path: Path, safe_size: int) -> np.ndarray:
    """加载原始地图切片并添加旋转安全边距。

    统一转换为 BGRA 四通道，以便后续旋转时用透明像素填充边缘，
    再经背景合成阶段将透明区域替换为真实背景。

    Args:
        path:      图片文件路径。
        safe_size: 旋转安全尺寸（由 get_safe_size() 计算）。

    Returns:
        添加了 pad 像素透明边距的 BGRA 图像，失败时返回 None。
    """
    img = safe_imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        logger.warning(f"Failed to load {path}")
        return None

    # 统一转为 BGRA，使旋转填充色（透明）与后续 Alpha 合成逻辑一致
    if img.ndim == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)

    pad = safe_size // 2
    return cv2.copyMakeBorder(
        img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=[0, 0, 0, 0]
    )


def extract_roi(img, cx: int, cy: int, angle: float, safe_size: int) -> np.ndarray:
    """以 (cx, cy) 为中心裁取并旋转 safe_size×safe_size 区域，返回 OUTPUT_SIZE 方块。

    Args:
        img:       已添加安全边距的图像（BGRA）。
        cx, cy:    采样中心坐标（相对于带边距图像）。
        angle:     旋转角度（度），0 表示不旋转。
        safe_size: 旋转安全尺寸。

    Returns:
        OUTPUT_SIZE×OUTPUT_SIZE 的裁剪结果。
    """
    half = safe_size // 2
    patch = img[cy - half : cy + half, cx - half : cx + half]

    if angle != 0:
        M = cv2.getRotationMatrix2D((half, half), angle, 1.0)
        border_val = (0, 0, 0, 0) if patch.shape[2] == 4 else (0, 0, 0)
        patch = cv2.warpAffine(patch, M, (safe_size, safe_size), borderValue=border_val)

    # 从旋转后的安全尺寸图像中心裁取目标尺寸
    start = (safe_size - CONFIG["OUTPUT_SIZE"]) // 2
    end = start + CONFIG["OUTPUT_SIZE"]
    return patch[start:end, start:end]


def is_valid(patch: np.ndarray) -> bool:
    """判断 Patch 是否包含足够丰富的地图纹理，过滤空白或单调区域。

    仅检验图像中心小窗口内的有效像素（灰度 > 5），要求：
      - 有效像素占比 ≥ 15%
      - 有效像素的灰度标准差 > STD_THRESHOLD

    Args:
        patch: OUTPUT_SIZE×OUTPUT_SIZE 的 BGRA 或 BGR 图像。

    Returns:
        True 表示纹理足够丰富，可作为训练样本。
    """
    bgr = patch[..., :3] if patch.shape[2] == 4 else patch
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # 仅在中心 60×60 区域评估，避免边缘黑框干扰判断
    cy, cx = CONFIG["OUTPUT_SIZE"] // 2, CONFIG["OUTPUT_SIZE"] // 2
    r = 30
    center = gray[cy - r : cy + r, cx - r : cx + r]
    valid_pixels = center[center > 5]

    if len(valid_pixels) < (center.size * 0.15):
        return False
    return np.std(valid_pixels) > CONFIG["STD_THRESHOLD"]


# ---------------------------------------------------------------------------
# 数据增强
# ---------------------------------------------------------------------------


def add_photometric_distortion(img: np.ndarray) -> np.ndarray:
    """随机光度畸变：高斯模糊 + HSV 色调/饱和度/亮度扰动。

    模拟不同时间段、光照条件下地图渲染的颜色差异，
    提升模型对光照变化的鲁棒性。

    Args:
        img: BGR 图像。

    Returns:
        增强后的 BGR 图像。
    """
    if random.random() > 0.7:
        img = cv2.GaussianBlur(img, (3, 3), random.uniform(0.5, 1.5))

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 0] = (hsv[:, :, 0] + random.uniform(-15, 15)) % 180
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * random.uniform(0.5, 1.5), 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * random.uniform(0.4, 1.5), 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def add_central_ui_simulation(img: np.ndarray) -> np.ndarray:
    """在图像中央叠加随机 UI 元素，仿真玩家头像与周边图标遮挡效果。

    游戏小地图中心始终存在玩家方向指示器，周围可能出现各类图标。
    此函数随机生成以下元素以提升模型遮挡鲁棒性：
      - 玩家中心光晕（模糊高亮圆）
      - 玩家方向三角指示器（随机朝向）
      - 周边随机图标：彩色方块、菱形串、雷达圆、角标框

    Args:
        img: BGR 图像（原地修改）。

    Returns:
        叠加 UI 元素后的 BGR 图像。
    """
    h, w = img.shape[:2]
    center = (w // 2, h // 2)

    # 玩家中心光晕
    glow_layer = np.zeros_like(img)
    glow_color = (random.randint(0, 50), random.randint(200, 255), 255)
    cv2.circle(glow_layer, center, random.randint(8, 14), glow_color, -1)
    glow_layer = cv2.GaussianBlur(glow_layer, (31, 31), 0)
    img = cv2.add(img, glow_layer)

    # 玩家方向三角指示器
    angle = random.uniform(0, 2 * math.pi)
    size = random.randint(10, 15)
    pt1 = (int(center[0] + size * math.cos(angle)), int(center[1] + size * math.sin(angle)))
    pt2 = (
        int(center[0] + size * 0.8 * math.cos(angle + 2.5)),
        int(center[1] + size * 0.8 * math.sin(angle + 2.5)),
    )
    pt3 = (
        int(center[0] + size * 0.8 * math.cos(angle - 2.5)),
        int(center[1] + size * 0.8 * math.sin(angle - 2.5)),
    )
    pts = np.array([pt1, pt2, pt3], np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(img, [pts], (255, 255, 255))
    cv2.polylines(img, [pts], isClosed=True, color=(40, 40, 40), thickness=1)

    # 周边随机图标（30% 概率跳过）
    if random.random() > 0.3:
        for _ in range(random.randint(1, 3)):
            offset_x = random.randint(-45, 45)
            offset_y = random.randint(-45, 45)
            icon_center = (center[0] + offset_x, center[1] + offset_y)
            icon_type = random.choice(
                ["color_block", "white_diamonds", "white_radar", "white_corners"]
            )

            if icon_type == "color_block":
                icon_color = random.choice(
                    [(255, 200, 50), (220, 220, 220), (255, 100, 100), (50, 200, 255)]
                )
                wh, hh = random.randint(5, 12), random.randint(5, 12)
                cv2.rectangle(
                    img,
                    (icon_center[0] - wh, icon_center[1] - hh),
                    (icon_center[0] + wh, icon_center[1] + hh),
                    icon_color,
                    -1,
                )
                cv2.rectangle(
                    img,
                    (icon_center[0] - wh, icon_center[1] - hh),
                    (icon_center[0] + wh, icon_center[1] + hh),
                    (255, 255, 255),
                    1,
                )

            elif icon_type == "white_diamonds":
                # 三颗垂直排列的菱形，最底部填充实心
                for dy in [-4, 0, 4]:
                    diamond_pts = np.array(
                        [
                            [icon_center[0], icon_center[1] - 4 + dy],
                            [icon_center[0] + 6, icon_center[1] + dy],
                            [icon_center[0], icon_center[1] + 4 + dy],
                            [icon_center[0] - 6, icon_center[1] + dy],
                        ],
                        np.int32,
                    )
                    cv2.polylines(img, [diamond_pts], isClosed=True, color=(255, 255, 255), thickness=1)
                    if dy == 4:
                        cv2.fillPoly(img, [diamond_pts], (255, 255, 255))

            elif icon_type == "white_radar":
                radius = random.randint(6, 11)
                cv2.circle(img, icon_center, radius, (255, 255, 255), 1)
                cv2.circle(img, icon_center, max(2, radius - 4), (200, 200, 200), -1)
                tri_pts = np.array(
                    [
                        [icon_center[0], icon_center[1] - 3],
                        [icon_center[0] + 3, icon_center[1] + 3],
                        [icon_center[0] - 3, icon_center[1] + 3],
                    ],
                    np.int32,
                )
                cv2.polylines(img, [tri_pts], isClosed=True, color=(255, 255, 255), thickness=1)

            elif icon_type == "white_corners":
                s = random.randint(15, 25)
                corner_pts = np.array(
                    [
                        [icon_center[0] - s // 2, icon_center[1] - s // 2 + 10],
                        [icon_center[0] - s // 2, icon_center[1] - s // 2],
                        [icon_center[0] - s // 2 + 10, icon_center[1] - s // 2],
                    ],
                    np.int32,
                )
                cv2.polylines(img, [corner_pts], isClosed=False, color=(255, 255, 255), thickness=1)

    return img


def apply_random_occlusion(patch: np.ndarray) -> np.ndarray:
    """在圆形有效区域内随机放置遮挡块，仿真 UI 元素遮挡地图的情况。

    遮挡块随机选择纯黑（50%）或彩色噪声（50%），位置限定在圆形掩码有效边界内。

    Args:
        patch: BGR 图像（原地修改）。

    Returns:
        添加随机遮挡后的图像。
    """
    # 限定遮挡范围在圆形区域内，避免在无效角落产生噪声
    offset = (CONFIG["OUTPUT_SIZE"] - CONFIG["MASK_DIAMETER"]) // 2
    limit = CONFIG["OUTPUT_SIZE"] - offset

    for _ in range(random.randint(1, CONFIG["OCCLUSION_COUNT"])):
        w = random.randint(10, CONFIG["OCCLUSION_SIZE"])
        h = random.randint(10, CONFIG["OCCLUSION_SIZE"])
        x = random.randint(offset, limit - w)
        y = random.randint(offset, limit - h)

        if random.random() > 0.5:
            patch[y : y + h, x : x + w] = 0
        else:
            patch[y : y + h, x : x + w] = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
    return patch


def augment_patch(patch: np.ndarray, safe_size: int) -> np.ndarray:
    """对单个 Patch 执行完整的增强流程。

    增强顺序：
        1. 光度畸变（50% 概率触发）
        2. 中心 UI 仿真（60% 概率触发）
        3. 圆形 Mask 应用（裁掉无效方形边角）
        4. 随机遮挡

    Args:
        patch:     BGR 图像（OUTPUT_SIZE×OUTPUT_SIZE）。
        safe_size: 当前旋转安全尺寸（接口保留，暂未使用）。

    Returns:
        增强后的 BGR 图像。
    """
    if random.random() > 0.5:
        patch = add_photometric_distortion(patch)

    if random.random() < 0.6:
        patch = add_central_ui_simulation(patch)

    # 应用圆形 Mask，使输出与推理端保持一致
    mask = np.zeros((CONFIG["OUTPUT_SIZE"], CONFIG["OUTPUT_SIZE"]), dtype=np.uint8)
    cv2.circle(
        mask,
        (CONFIG["OUTPUT_SIZE"] // 2, CONFIG["OUTPUT_SIZE"] // 2),
        CONFIG["MASK_DIAMETER"] // 2,
        255,
        -1,
    )
    patch = cv2.bitwise_and(patch, patch, mask=mask)

    return apply_random_occlusion(patch)


# ---------------------------------------------------------------------------
# 背景合成
# ---------------------------------------------------------------------------

# 模块级背景缓存，避免多进程子进程重复加载同一张大图
_bg_cache: dict = {}


def apply_background_composition(patch_bgra: np.ndarray, bg_paths: list) -> np.ndarray:
    """将带 Alpha 通道的 Patch 与随机背景进行 Alpha 合成。

    背景选取策略（按概率分配）：
        60%：从 bg_paths 中随机选取真实背景图裁块
        15%：纯黑底（防止灾难性遗忘）
        15%：纯白/浅灰底（适应过曝或 UI 遮挡场景）
        10%：随机彩色噪点底（极限泛化）

    背景缓存限制原始图像最长边为 400px，防止多进程下高清大图驻留内存导致 OOM。

    Args:
        patch_bgra: BGRA 格式的源 Patch（透明通道表示地图有效区域）。
        bg_paths:   可用背景图路径列表；为空时退化为纯黑底。

    Returns:
        Alpha 合成完毕的 BGR 图像。
    """
    if patch_bgra.shape[2] != 4:
        return patch_bgra

    h, w = patch_bgra.shape[:2]
    bgr = patch_bgra[..., :3].astype(np.float32)
    alpha = (patch_bgra[..., 3] / 255.0).astype(np.float32)
    alpha = np.expand_dims(alpha, axis=-1)

    rand_val = random.random()

    if rand_val < 0.6 and bg_paths:
        bg_path_str = str(random.choice(bg_paths))

        if bg_path_str not in _bg_cache:
            loaded_bg = safe_imread(bg_path_str)
            if loaded_bg is not None:
                bh, bw = loaded_bg.shape[:2]
                max_dim = 400
                if bh > max_dim or bw > max_dim:
                    scale = max_dim / max(bh, bw)
                    loaded_bg = cv2.resize(loaded_bg, (int(bw * scale), int(bh * scale)))
            _bg_cache[bg_path_str] = loaded_bg

        bg_source = _bg_cache[bg_path_str]

        if bg_source is not None:
            bg_h, bg_w = bg_source.shape[:2]

            # 若背景图小于 Patch，则等比放大至可覆盖
            if bg_h < h or bg_w < w:
                scale = max(h / bg_h, w / bg_w)
                new_w, new_h = int(bg_w * scale) + 1, int(bg_h * scale) + 1
                bg_source = cv2.resize(bg_source, (new_w, new_h))
                bg_h, bg_w = new_h, new_w

            y = random.randint(0, max(0, bg_h - h))
            x = random.randint(0, max(0, bg_w - w))
            bg_patch = bg_source[y : y + h, x : x + w].astype(np.float32)

            if bg_patch.shape[:2] != (h, w):
                bg_patch = cv2.resize(bg_patch, (w, h))
        else:
            bg_patch = np.zeros((h, w, 3), dtype=np.float32)

    elif rand_val < 0.75:
        bg_patch = np.zeros((h, w, 3), dtype=np.float32)

    elif rand_val < 0.90:
        gray_val = random.randint(200, 255)
        bg_patch = np.full((h, w, 3), gray_val, dtype=np.float32)

    else:
        # 用低分辨率噪声放大：生成速度快且视觉效果接近随机纹理
        small_noise = np.random.randint(0, 256, (16, 16, 3), dtype=np.uint8)
        bg_patch = cv2.resize(small_noise, (w, h), interpolation=cv2.INTER_NEAREST).astype(
            np.float32
        )

    result = bgr * alpha + bg_patch * (1.0 - alpha)
    return result.astype(np.uint8)


# ---------------------------------------------------------------------------
# 样本生成
# ---------------------------------------------------------------------------


def process_patch(
    img: np.ndarray, cx: int, cy: int, angle: float, safe_size: int, bg_paths: list
) -> np.ndarray:
    """提取单个位置/角度的样本并完成背景合成与增强。"""
    patch_bgra = extract_roi(img, cx, cy, angle, safe_size)
    patch_bgr = apply_background_composition(patch_bgra, bg_paths)
    return augment_patch(patch_bgr, safe_size)


def load_error_images(class_name: str, error_dir: Path) -> list:
    """加载并过采样指定类别的困难负样本。

    通过翻转和随机遮挡对每张困难样本生成 ERROR_OVERSAMPLE 份变体，
    将模型在易混淆区域的误识率纳入训练目标。

    Args:
        class_name: 类别名称，对应 error_dir 下的子目录名。
        error_dir:  困难负样本根目录。

    Returns:
        过采样后的 BGR 图像列表。
    """
    error_samples = []
    error_path = error_dir / class_name

    for ext in ("*.[pP][nN][gG]", "*.[jJ][pP][gG]"):
        for file_path in error_path.glob(ext):
            img = safe_imread(file_path)
            if img is None:
                continue

            if img.shape[:2] != (CONFIG["OUTPUT_SIZE"], CONFIG["OUTPUT_SIZE"]):
                img = cv2.resize(img, (CONFIG["OUTPUT_SIZE"], CONFIG["OUTPUT_SIZE"]))

            for _ in range(CONFIG["ERROR_OVERSAMPLE"]):
                aug = img.copy()
                if random.random() > 0.5:
                    aug = cv2.flip(aug, random.choice([-1, 0, 1]))
                aug = apply_random_occlusion(aug)
                error_samples.append(aug)

    return error_samples


def generate_samples(img: np.ndarray, safe_size: int, bg_paths: list) -> list:
    """从单张地图切片生成 TARGET_COUNT 个增强样本。

    分两阶段采样：
        第一阶段：滑窗扫描，0° 旋转，收集所有有效中心坐标并直接采样。
        第二阶段：若数量不足，从有效中心随机选取，叠加位置扰动（±5px）
                  和随机旋转角，直至达到目标数量。

    Args:
        img:       已添加安全边距的 BGRA 图像。
        safe_size: 旋转安全尺寸。
        bg_paths:  背景图路径列表。

    Returns:
        BGR 图像列表，长度约等于 TARGET_COUNT。
    """
    h, w = img.shape[:2]
    pad = safe_size // 2
    orig_h, orig_w = h - 2 * pad, w - 2 * pad

    # 阶段一：滑窗扫描，筛选有效中心
    valid_centers = []
    for y in range(0, orig_h, CONFIG["STRIDE"]):
        for x in range(0, orig_w, CONFIG["STRIDE"]):
            cx, cy = x + pad, y + pad
            if is_valid(extract_roi(img, cx, cy, 0, safe_size)):
                valid_centers.append((cx, cy))

    if not valid_centers:
        return []

    samples = []
    for cx, cy in valid_centers:
        if len(samples) >= CONFIG["TARGET_COUNT"]:
            break
        samples.append(process_patch(img, cx, cy, 0, safe_size, bg_paths))

    # 阶段二：随机旋转扩充至 TARGET_COUNT
    while len(samples) < CONFIG["TARGET_COUNT"]:
        cx, cy = random.choice(valid_centers)
        nx = max(pad, min(cx + random.randint(-5, 5), w - pad))
        ny = max(pad, min(cy + random.randint(-5, 5), h - pad))
        angle = random.uniform(0, 360)

        patch = extract_roi(img, nx, ny, angle, safe_size)
        if is_valid(patch):
            if random.random() > 0.5:
                patch = cv2.flip(patch, random.choice([-1, 0, 1]))
            patch_bgr = apply_background_composition(patch, bg_paths)
            samples.append(augment_patch(patch_bgr, safe_size))

    return samples


def save_dataset(samples: list, class_name: str, file_stem: str, output_dir: Path) -> None:
    """将样本列表按 VAL_RATIO 随机划分并写入 train/val 目录。

    Args:
        samples:    BGR 图像列表。
        class_name: 类别名称（对应输出子目录名）。
        file_stem:  源文件名（不含扩展名），用于生成输出文件名前缀。
        output_dir: 数据集根目录。
    """
    random.shuffle(samples)
    split_idx = int(len(samples) * CONFIG["VAL_RATIO"])

    val_samples = samples[:split_idx]
    train_samples = samples[split_idx:]

    train_class_dir = output_dir / "train" / class_name
    val_class_dir = output_dir / "val" / class_name
    train_class_dir.mkdir(parents=True, exist_ok=True)
    val_class_dir.mkdir(parents=True, exist_ok=True)

    for i, img in enumerate(train_samples):
        safe_imwrite(train_class_dir / f"{file_stem}_{i:05d}.jpg", img)

    for i, img in enumerate(val_samples):
        safe_imwrite(val_class_dir / f"{file_stem}_{i:05d}.jpg", img)


# ---------------------------------------------------------------------------
# 并行任务入口
# ---------------------------------------------------------------------------


def process_image_task(
    file_path: Path,
    class_name: str,
    output_dir: Path,
    error_dir: Path,
    bg_paths: list,
) -> str:
    """单张图片的完整处理流程，供 ProcessPoolExecutor 调度。

    Args:
        file_path:  输入图片路径。
        class_name: 所属类别名称。
        output_dir: 数据集根目录。
        error_dir:  困难负样本目录（可为 None）。
        bg_paths:   背景图路径列表。

    Returns:
        描述处理结果的字符串，用于主进程日志输出。
    """
    safe_size = get_safe_size()

    img = load_image(file_path, safe_size)
    if img is None:
        return f"Failed to load {file_path}"

    samples = generate_samples(img, safe_size, bg_paths)

    if error_dir and (error_dir / class_name).exists():
        error_samples = load_error_images(class_name, error_dir)
        if error_samples:
            samples.extend(error_samples)

    save_dataset(samples, class_name, file_path.stem, output_dir)
    return f"Completed {class_name} ({file_path.name})"


# ---------------------------------------------------------------------------
# 主流水线
# ---------------------------------------------------------------------------


class DataPreprocessor:
    """数据集生成主控类。

    负责：目录扫描、任务调度（多进程并行）、进度汇报。

    Args:
        input_dir:  原始地图切片根目录。
        output_dir: 数据集输出根目录（运行前自动清空）。
        error_dir:  困难负样本目录（可选，字符串或空字符串）。
        bg_dir:     背景域随机化图片目录（可选，字符串或空字符串）。
    """

    def __init__(self, input_dir: str, output_dir: str, error_dir: str, bg_dir: str):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.error_dir = Path(error_dir) if error_dir else None

        self.bg_paths: list[Path] = []
        bg_dir_path = Path(bg_dir) if bg_dir else None
        if bg_dir_path and bg_dir_path.exists():
            for ext in ["*.[pP][nN][gG]", "*.[jJ][pP][gG]"]:
                self.bg_paths.extend(list(bg_dir_path.glob(ext)))
        logger.info(f"Discovered {len(self.bg_paths)} background images.")

    def run(self) -> None:
        """执行完整的数据集生成流水线。"""
        # 清空旧输出，保证数据集干净
        if self.output_dir.exists():
            shutil.rmtree(self.output_dir)

        (self.output_dir / "train").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "val").mkdir(parents=True, exist_ok=True)

        tasks: list[tuple[Path, str]] = []

        # 根目录下的图片：文件名（不含扩展名）作为类别名
        for ext in ["*.[pP][nN][gG]", "*.[jJ][pP][gG]"]:
            for file_path in self.input_dir.glob(ext):
                tasks.append((file_path, file_path.stem))

        # 子目录：目录名作为类别名
        for child in self.input_dir.iterdir():
            if not child.is_dir():
                continue
            class_name = child.name
            if class_name.startswith(".") or class_name == "__pycache__":
                continue

            logger.info(f"Queuing class directory: {class_name}")
            for ext in ["*.[pP][nN][gG]", "*.[jJ][pP][gG]"]:
                for file_path in child.glob(ext):
                    tasks.append((file_path, class_name))

        max_workers = os.cpu_count() or 4
        logger.info(
            f"Starting parallel processing with {max_workers} workers for {len(tasks)} tasks..."
        )

        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {
                executor.submit(
                    process_image_task,
                    file_path,
                    class_name,
                    self.output_dir,
                    self.error_dir,
                    self.bg_paths,
                ): file_path
                for file_path, class_name in tasks
            }

            for future in concurrent.futures.as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    logger.info(future.result())
                except Exception as e:
                    logger.error(f"Error processing {file_path}: {e}")

        logger.info("Preprocessing completed successfully.")


# ---------------------------------------------------------------------------
# CLI 入口
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import multiprocessing

    # Windows 下多进程必须在 if __name__ == "__main__" 保护块内调用
    multiprocessing.freeze_support()

    parser = argparse.ArgumentParser(description="Dataset Preprocessing Pipeline")
    parser.add_argument("--input", default="source_images", help="Input directory containing source map images")
    parser.add_argument("--output", default="dataset", help="Output directory for the generated dataset")
    parser.add_argument(
        "--error",
        default="error_images",
        help="Directory containing hard negative samples for error mining",
    )
    parser.add_argument(
        "--bg",
        default="bg_images",
        help="Directory containing background images for domain randomization",
    )
    args = parser.parse_args()

    DataPreprocessor(args.input, args.output, args.error, args.bg).run()
