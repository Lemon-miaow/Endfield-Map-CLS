"""
preprocess.py — 训练数据集生成流水线

从 source_images 生成 YOLO 分类训练集，覆盖普通地图类、Base 大图 tile 类、
None 类负样本、困难样本过采样和 UI/背景域随机化。默认配置直接运行即可，
命令行参数只作为覆盖项。

目录约定:
    source_images/                  原始地图图片根目录
        <class_name>.png            根目录图片的文件名即类别名
        <class_name>/               子目录图片使用目录名作为类别名
            *.png / *.jpg

    error_images/<class_name>/      困难样本目录；图片会按 ERROR_OVERSAMPLE 并入训练集
    bg_images/                      背景域随机化图片目录（可选）
    dataset/                        输出数据集根目录，运行前自动清空

用法:
    python preprocess.py [--input <dir>] [--output <dir>]
                         [--error <dir>] [--bg <dir>]
                         [--target-count <int>] [--workers <int>]
"""

from __future__ import annotations

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
# 训练数据默认配置；CLI 只覆盖这些默认值。
# ---------------------------------------------------------------------------

CONFIG = {
    "OUTPUT_SIZE": 128,                # 输出训练图像边长
    "MASK_DIAMETER": 106,              # 小地图圆形有效区域直径
    "TARGET_COUNT": 1200,              # 普通类别目标样本数
    "TIER_TARGET_COUNT": 300,          # Tier 类目标样本数
    "NONE_CLASS_TOTAL_CAP": 3000,      # None 类总样本上限
    "NONE_PER_IMAGE_MIN": 40,          # 单张 None 图最小采样数
    "NONE_PER_IMAGE_MAX": 100,         # 单张 None 图最大采样数
    "VAL_RATIO": 0.2,                  # 验证集比例
    "STRIDE": 8,                       # 滑窗扫描步长
    "ANGLE_JITTER": 0.5,               # 随机旋转扰动角度
    "STD_THRESHOLD": 5.0,              # 保留兼容字段，实际有效性使用 MIN_VALID_STD
    "OCCLUSION_COUNT": 0,              # 保留兼容字段
    "OCCLUSION_SIZE": 0,               # 保留兼容字段
    "ERROR_OVERSAMPLE": 5,             # 困难样本过采样倍数
    "BACKGROUND_BLEND_RANGE": (0.9, 1.0),
    "BASE_CLASS_NAMES": {"Map01Base", "Map02Base"},
    "TILE_SIZE": 160,
    "TILE_STRIDE": 160,
    "TILE_INFER_MARGIN": 64,
    "UI_ICON_SCALE": 0.08,
    "UI_BLUE_PROB": 0.5,
    "UI_ICON_MIN_COUNT": 2,
    "UI_ICON_MAX_COUNT": 6,
    "EXTREME_UI_PROB": 0.06,
    "EXTREME_UI_PACK_MIN": 1,
    "EXTREME_UI_PACK_MAX": 2,
    "EXTREME_UI_ICONS_MIN": 8,
    "EXTREME_UI_ICONS_MAX": 16,
    "EXTREME_UI_RADIUS_MIN": 10,
    "EXTREME_UI_RADIUS_MAX": 22,
    "EXTREME_UI_CHAIN_PROB": 0.5,
    "EXTREME_UI_EDGE_BIAS_PROB": 0.35,
    "UI_POINTER_PROB": 1.0,
    "MIN_MAP_CIRCLE_COVERAGE": 0.10,
    "MIN_MAP_CENTER_COVERAGE": 0.035,
    "MIN_ALPHA_CIRCLE_COVERAGE": 0.12,
    "MIN_VALID_STD": 8.0,
    "MIN_VALID_CENTERS_PER_TILE": 24,
}

DEFAULT_OPTIONS = {
    "input": "source_images",
    "output": "dataset",
    "error": "error_images",
    "bg": "bg_images",
    "workers": None,
}

CONFIG_OVERRIDES = {
    "target_count": "TARGET_COUNT",
    "tier_target_count": "TIER_TARGET_COUNT",
    "none_total_cap": "NONE_CLASS_TOTAL_CAP",
    "none_per_image_min": "NONE_PER_IMAGE_MIN",
    "none_per_image_max": "NONE_PER_IMAGE_MAX",
    "val_ratio": "VAL_RATIO",
    "stride": "STRIDE",
    "error_oversample": "ERROR_OVERSAMPLE",
}

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 基础 I/O 工具
# ---------------------------------------------------------------------------


def safe_imread(path, flags=cv2.IMREAD_COLOR):
    """读取图片，兼容 Path 对象和非 ASCII 文件名。"""
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), flags)


def safe_imwrite(path, img):
    """写入图片，兼容 Path 对象和非 ASCII 文件名。"""
    ext = os.path.splitext(str(path))[1]
    is_success, buf = cv2.imencode(ext, img)
    if is_success:
        buf.tofile(str(path))
    return is_success


# ---------------------------------------------------------------------------
# UI 图标处理工具
# ---------------------------------------------------------------------------

_ui_icon_cache = None


def load_ui_icons(icon_dir: str) -> dict:
    """加载并缓存地图 UI 图标的绘制变体。"""
    p = Path(icon_dir)
    if not p.exists():
        logger.warning(f"Icon directory {icon_dir} does not exist.")
        return {}

    raw_icons = {}
    for file_path in p.glob("*.png"):
        img = safe_imread(file_path, cv2.IMREAD_UNCHANGED)
        if img is None or img.ndim != 3 or img.shape[2] != 4:
            continue
        raw_icons[file_path.name] = img

    processed = {"normal": {}, "pointer": None}

    for name, img in raw_icons.items():
        if name.lower() == "pointer.png":
            processed["pointer"] = sanitize_rgba_alpha(img, alpha_floor=8)
        else:
            sanitized = sanitize_rgba_alpha(img, alpha_floor=8)

            white_outlined = add_black_outline_rgba(
                sanitized,
                thickness=2,
                alpha_threshold=32,
            )
            blue_tinted = tint_icon_blue_rgba(sanitized, (248, 205, 97))
            blue_outlined = add_black_outline_rgba(
                blue_tinted,
                thickness=2,
                alpha_threshold=32,
            )

            processed["normal"][name] = {"white": white_outlined, "blue": blue_outlined}

    logger.info(f"Loaded and pre-processed UI icons: {sorted(raw_icons.keys())}")
    if processed["pointer"] is None:
        logger.warning("pointer.png not found in icon directory.")

    return processed


def sanitize_rgba_alpha(icon_rgba: np.ndarray, alpha_floor: int = 8) -> np.ndarray:
    result = icon_rgba.copy()
    low_alpha = result[..., 3] < alpha_floor
    result[low_alpha, 0] = 0
    result[low_alpha, 1] = 0
    result[low_alpha, 2] = 0
    result[low_alpha, 3] = 0
    return result


def add_black_outline_rgba(
    icon_rgba: np.ndarray,
    thickness: int = 1,
    alpha_threshold: int = 32,
) -> np.ndarray:
    alpha = icon_rgba[..., 3]
    solid = alpha > alpha_threshold

    kernel = np.ones((thickness * 2 + 1, thickness * 2 + 1), np.uint8)
    dilated = cv2.dilate(solid.astype(np.uint8) * 255, kernel, iterations=1) > 0
    outline = dilated & (~solid)

    outline_layer = np.zeros_like(icon_rgba)
    outline_layer[outline, 0] = 0
    outline_layer[outline, 1] = 0
    outline_layer[outline, 2] = 0
    outline_layer[outline, 3] = 255

    src = icon_rgba.astype(np.float32)
    dst = outline_layer.astype(np.float32)

    src_a = src[..., 3:4] / 255.0
    dst_a = dst[..., 3:4] / 255.0

    out_a = src_a + dst_a * (1.0 - src_a)
    out_rgb = src[..., :3] * src_a + dst[..., :3] * dst_a * (1.0 - src_a)

    safe_a = np.maximum(out_a, 1e-6)
    out_rgb = out_rgb / safe_a

    result = np.zeros_like(icon_rgba)
    result[..., :3] = np.clip(out_rgb, 0, 255).astype(np.uint8)
    result[..., 3] = np.clip(out_a[..., 0] * 255, 0, 255).astype(np.uint8)

    return result


def tint_icon_blue_rgba(icon_rgba: np.ndarray, bgr_color=(248, 205, 97)) -> np.ndarray:
    result = icon_rgba.copy()
    alpha = result[..., 3] > 0
    result[alpha, 0] = bgr_color[0]
    result[alpha, 1] = bgr_color[1]
    result[alpha, 2] = bgr_color[2]
    return result


def resize_rgba(icon_rgba: np.ndarray, scale: float, min_side: int | None = None) -> np.ndarray:
    h, w = icon_rgba.shape[:2]
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    if min_side is not None and min(new_w, new_h) < min_side:
        factor = min_side / max(1, min(new_w, new_h))
        new_w = int(round(new_w * factor))
        new_h = int(round(new_h * factor))

    return cv2.resize(icon_rgba, (new_w, new_h), interpolation=cv2.INTER_AREA)


def overlay_rgba_on_bgr(dst_bgr: np.ndarray, icon_rgba: np.ndarray, x: int, y: int) -> np.ndarray:
    h, w = dst_bgr.shape[:2]
    ih, iw = icon_rgba.shape[:2]

    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(w, x + iw)
    y2 = min(h, y + ih)

    if x1 >= x2 or y1 >= y2:
        return dst_bgr

    sx1 = x1 - x
    sy1 = y1 - y
    sx2 = sx1 + (x2 - x1)
    sy2 = sy1 + (y2 - y1)

    src = icon_rgba[sy1:sy2, sx1:sx2].astype(np.float32)
    alpha = src[..., 3:4] / 255.0
    src_bgr = src[..., :3]

    dst = dst_bgr[y1:y2, x1:x2].astype(np.float32)
    blended = src_bgr * alpha + dst * (1.0 - alpha)
    dst_bgr[y1:y2, x1:x2] = blended.astype(np.uint8)

    return dst_bgr


def _sample_normal_ui_icon(normal_icons: dict, icon_names: list[str], name: str | None = None) -> np.ndarray:
    if name is None:
        name = random.choice(icon_names)

    variants = normal_icons[name]
    mode = "blue" if random.random() < CONFIG["UI_BLUE_PROB"] else "white"
    return resize_rgba(variants[mode], CONFIG["UI_ICON_SCALE"])


def draw_one_normal_icon(result: np.ndarray, icon_rgba: np.ndarray, x: int, y: int) -> np.ndarray:
    return overlay_rgba_on_bgr(result, icon_rgba, x, y)


def sample_extreme_anchor(h: int, w: int) -> tuple[int, int]:
    """采样极端 UI 干扰的锚点，优先覆盖边缘高风险区域。"""
    if random.random() < CONFIG["EXTREME_UI_EDGE_BIAS_PROB"]:
        if random.random() < 0.6:
            x_low = max(20, int(w * 0.55))
            x_high = max(x_low, w - 20)
            y_low = 20
            y_high = max(y_low, int(h * 0.42))
            return random.randint(x_low, x_high), random.randint(y_low, y_high)

        side = random.choice(["top", "bottom", "left", "right"])
        if side == "top":
            x_low, x_high = 20, max(20, w - 20)
            y_low, y_high = 20, max(20, min(38, h - 20))
            return random.randint(x_low, x_high), random.randint(y_low, y_high)
        if side == "bottom":
            x_low, x_high = 20, max(20, w - 20)
            y_low, y_high = max(20, h - 38), max(max(20, h - 38), h - 20)
            return random.randint(x_low, x_high), random.randint(y_low, y_high)
        if side == "left":
            x_low, x_high = 20, max(20, min(38, w - 20))
            y_low, y_high = 20, max(20, h - 20)
            return random.randint(x_low, x_high), random.randint(y_low, y_high)
        x_low, x_high = max(20, w - 38), max(max(20, w - 38), w - 20)
        y_low, y_high = 20, max(20, h - 20)
        return random.randint(x_low, x_high), random.randint(y_low, y_high)

    x_low, x_high = 24, max(24, w - 24)
    y_low, y_high = 24, max(24, h - 24)
    return random.randint(x_low, x_high), random.randint(y_low, y_high)


def add_extreme_icon_clutter(result: np.ndarray, normal_icons: dict, icon_names: list[str]) -> np.ndarray:
    """在局部区域叠加高密度图标，模拟强遮挡场景。"""
    if not icon_names:
        return result

    h, w = result.shape[:2]
    out = result.copy()

    pack_count = random.randint(CONFIG["EXTREME_UI_PACK_MIN"], CONFIG["EXTREME_UI_PACK_MAX"])

    for _ in range(pack_count):
        cx, cy = sample_extreme_anchor(h, w)
        radius = random.randint(CONFIG["EXTREME_UI_RADIUS_MIN"], CONFIG["EXTREME_UI_RADIUS_MAX"])
        icon_count = random.randint(CONFIG["EXTREME_UI_ICONS_MIN"], CONFIG["EXTREME_UI_ICONS_MAX"])

        if random.random() < CONFIG["EXTREME_UI_CHAIN_PROB"]:
            angle = random.uniform(0, 2 * np.pi)
            step = random.randint(4, 8)

            for i in range(icon_count):
                t = (i - (icon_count - 1) / 2.0) * step
                dx = int(round(np.cos(angle) * t + random.gauss(0, 2.0)))
                dy = int(round(np.sin(angle) * t + random.gauss(0, 2.0)))

                icon = _sample_normal_ui_icon(normal_icons, icon_names)
                ih, iw = icon.shape[:2]
                x = min(max(cx + dx - iw // 2, 0), max(0, w - iw))
                y = min(max(cy + dy - ih // 2, 0), max(0, h - ih))

                out = draw_one_normal_icon(out, icon, x, y)
        else:
            for _ in range(icon_count):
                dx = int(round(random.gauss(0, radius * 0.45)))
                dy = int(round(random.gauss(0, radius * 0.45)))

                icon = _sample_normal_ui_icon(normal_icons, icon_names)
                ih, iw = icon.shape[:2]
                x = min(max(cx + dx - iw // 2, 0), max(0, w - iw))
                y = min(max(cy + dy - ih // 2, 0), max(0, h - ih))

                out = draw_one_normal_icon(out, icon, x, y)

    return out


def draw_random_ui_lines(img: np.ndarray) -> np.ndarray:
    """绘制轻量路线线条，模拟小地图上的路径 UI。"""
    h, w = img.shape[:2]
    overlay = img.copy()

    if random.random() < 0.7:
        num_points = random.randint(2, 4)
        pts = np.array(
            [[random.randint(8, w - 8), random.randint(8, h - 8)] for _ in range(num_points)],
            np.int32
        )
        cv2.polylines(overlay, [pts], False, (0, 255, 255), 1)

    if random.random() < 0.7:
        num_points = random.randint(2, 5)
        pts = np.array(
            [[random.randint(8, w - 8), random.randint(8, h - 8)] for _ in range(num_points)],
            np.int32
        )
        cv2.polylines(overlay, [pts], False, (255, 255, 255), 1)

    return overlay


def add_real_ui_icons(img: np.ndarray, icon_dir: str = "icon") -> np.ndarray:
    """叠加真实 UI 图标和中心指针，保持与游戏小地图遮挡形态接近。"""
    global _ui_icon_cache
    if _ui_icon_cache is None:
        _ui_icon_cache = load_ui_icons(icon_dir)

    if not _ui_icon_cache:
        return img

    h, w = img.shape[:2]
    result = img.copy()

    normal_icons = _ui_icon_cache["normal"]
    icon_names = list(normal_icons.keys())
    if icon_names:
        use_count = random.randint(CONFIG["UI_ICON_MIN_COUNT"], CONFIG["UI_ICON_MAX_COUNT"])

        for _ in range(use_count):
            icon = _sample_normal_ui_icon(normal_icons, icon_names)

            ih, iw = icon.shape[:2]
            x = random.randint(0, max(0, w - iw))
            y = random.randint(0, max(0, h - ih))

            result = draw_one_normal_icon(result, icon, x, y)

        if random.random() < CONFIG["EXTREME_UI_PROB"]:
            result = add_extreme_icon_clutter(result, normal_icons, icon_names)

    pointer_icon = _ui_icon_cache["pointer"]
    if pointer_icon is not None:
        angle = random.uniform(0, 360)
        ih, iw = pointer_icon.shape[:2]
        center = (iw // 2, ih // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_pointer = cv2.warpAffine(
            pointer_icon,
            M,
            (iw, ih),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0),
        )

        ih, iw = rotated_pointer.shape[:2]
        x = (w - iw) // 2
        y = (h - ih) // 2

        result = overlay_rgba_on_bgr(result, rotated_pointer, x, y)

    return result


def debug_pointer_alpha(icon_dir: str = "icon") -> None:
    pointer_path = Path(icon_dir) / "pointer.png"
    img = safe_imread(pointer_path, cv2.IMREAD_UNCHANGED)

    if img is None:
        logger.warning("pointer.png not found.")
        return

    alpha = img[..., 3]
    logger.info(
        f"pointer alpha: shape={img.shape}, "
        f"min={alpha.min()}, max={alpha.max()}, "
        f"nonzero={(alpha > 0).sum()}, "
        f"solid={(alpha > 32).sum()}"
    )

    safe_imwrite("debug_pointer_alpha.png", alpha)


def build_axis_positions(length: int, tile_size: int, stride: int) -> list[int]:
    """从左上角开始按固定 stride 铺网格。

    不再使用“贴边补 last”的逻辑。
    最后一个 tile 如果超出边界，会在 enumerate_base_tiles() 中截断到图像大小。
    """
    if length <= 0:
        return [0]
    return list(range(0, length, stride))


def enumerate_base_tiles(orig_h: int, orig_w: int, class_name: str) -> list[dict]:
    """为 Base 大图生成固定 stride 网格。

    规则：
    - 左上角为起点
    - 按 TILE_STRIDE 正常铺开
    - 每个 tile 的理论大小为 TILE_SIZE x TILE_SIZE
    - 若超出原图边界，则 x2/y2 截断到实际图像大小
    - 宽高小于等于 0 的无效块直接跳过
    """
    tile_size = CONFIG["TILE_SIZE"]
    stride = CONFIG["TILE_STRIDE"]

    ys = build_axis_positions(orig_h, tile_size, stride)
    xs = build_axis_positions(orig_w, tile_size, stride)

    tiles = []
    for r, y in enumerate(ys):
        for c, x in enumerate(xs):
            x1 = x
            y1 = y
            x2 = min(x + tile_size, orig_w)
            y2 = min(y + tile_size, orig_h)
            w = x2 - x1
            h = y2 - y1
            if w <= 0 or h <= 0:
                continue

            tiles.append(
                {
                    "class_name": f"{class_name}__r{r:02d}_c{c:02d}",
                    "base_class": class_name,
                    "row": r,
                    "col": c,
                    "x": x1,
                    "y": y1,
                    "w": w,
                    "h": h,
                }
            )
    return tiles


def save_tile_mapping(mapping: dict, output_dir: Path) -> None:
    """保存 Base tile 到原图区域的映射表。"""
    import json

    mapping_path = output_dir / "tile_mapping.json"
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# 图像预处理
# ---------------------------------------------------------------------------


def get_safe_size() -> int:
    """计算旋转后仍可中心裁剪 OUTPUT_SIZE 的安全边长。"""
    return int(math.ceil(math.sqrt(2 * CONFIG["OUTPUT_SIZE"] ** 2)))


def load_image(path: Path, safe_size: int) -> np.ndarray:
    """加载源图并补透明边距，供后续旋转裁剪。"""
    img = safe_imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        logger.warning(f"Failed to load {path}")
        return None

    if img.ndim == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)

    pad = safe_size // 2
    return cv2.copyMakeBorder(
        img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=[0, 0, 0, 0]
    )


def extract_roi(img, cx: int, cy: int, angle: float, safe_size: int) -> np.ndarray:
    """以中心点裁出 ROI，按需旋转后返回训练尺寸图块。"""
    half = safe_size // 2
    patch = img[cy - half : cy + half, cx - half : cx + half]

    if angle != 0:
        M = cv2.getRotationMatrix2D((half, half), angle, 1.0)
        border_val = (0, 0, 0, 0) if patch.shape[2] == 4 else (0, 0, 0)
        patch = cv2.warpAffine(patch, M, (safe_size, safe_size), borderValue=border_val)

    start = (safe_size - CONFIG["OUTPUT_SIZE"]) // 2
    end = start + CONFIG["OUTPUT_SIZE"]
    return patch[start:end, start:end]


def is_valid(patch: np.ndarray) -> bool:
    """判断图块是否包含足够地图内容，过滤空洞和碎片区域。"""
    size = CONFIG["OUTPUT_SIZE"]
    radius = CONFIG["MASK_DIAMETER"] // 2
    cy, cx = size // 2, size // 2

    circle_mask = np.zeros((size, size), dtype=np.uint8)
    cv2.circle(circle_mask, (cx, cy), radius, 255, -1)
    circle_bool = circle_mask > 0
    circle_area = max(1, int(np.count_nonzero(circle_bool)))

    if patch.shape[2] == 4:
        raw_alpha = patch[..., 3]
        alpha_bool = raw_alpha > 10

        bgr = patch[..., :3].astype(np.float32)
        alpha = (raw_alpha / 255.0).astype(np.float32)[..., None]
        bgr = (bgr * alpha).astype(np.uint8)

        alpha_circle_ratio = np.count_nonzero(alpha_bool & circle_bool) / circle_area
        if alpha_circle_ratio < CONFIG["MIN_ALPHA_CIRCLE_COVERAGE"]:
            return False
    else:
        bgr = patch.copy()
        alpha_bool = np.ones((size, size), dtype=bool)

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    map_bool = (gray > 12) & alpha_bool & circle_bool

    map_circle_ratio = np.count_nonzero(map_bool) / circle_area
    if map_circle_ratio < CONFIG["MIN_MAP_CIRCLE_COVERAGE"]:
        return False

    r = 30
    center_bool = np.zeros((size, size), dtype=bool)
    center_bool[cy - r : cy + r, cx - r : cx + r] = True

    center_area = max(1, int(np.count_nonzero(center_bool & circle_bool)))
    map_center_ratio = np.count_nonzero(map_bool & center_bool) / center_area
    if map_center_ratio < CONFIG["MIN_MAP_CENTER_COVERAGE"]:
        return False

    valid_pixels = gray[map_bool]
    if valid_pixels.size < 80:
        return False

    if float(np.std(valid_pixels)) < CONFIG["MIN_VALID_STD"]:
        return False

    component_mask = map_bool.astype(np.uint8) * 255
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(component_mask, 8)

    if num_labels <= 1:
        return False

    largest_area = int(stats[1:, cv2.CC_STAT_AREA].max())
    if largest_area < 120:
        return False

    return True


# ---------------------------------------------------------------------------
# 数据增强
# ---------------------------------------------------------------------------


def add_photometric_distortion(img: np.ndarray) -> np.ndarray:
    """执行轻量光照和模糊扰动。"""
    if random.random() < 0.15:
        alpha = random.uniform(0.97, 1.03)
        beta = random.uniform(-2, 2)
        img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    if random.random() < 0.03:
        img = cv2.GaussianBlur(img, (3, 3), random.uniform(0.2, 0.4))

    return img


def apply_random_occlusion(patch: np.ndarray) -> np.ndarray:
    """保留旧接口；当前验证配置不启用额外遮挡。"""
    return patch


def apply_synthetic_fog(patch: np.ndarray, feature_strength: float) -> np.ndarray:
    """保留旧接口；当前验证配置不启用合成雾化。"""
    return patch


def add_central_ui_simulation(img: np.ndarray, feature_strength: float | None = None) -> np.ndarray:
    """叠加中心 UI 干扰。"""
    result = img.copy()
    result = draw_random_ui_lines(result)
    result = add_real_ui_icons(result, "icon")
    return result


def augment_patch(patch: np.ndarray, safe_size: int) -> np.ndarray:
    """对普通类别样本执行完整增强。"""
    if random.random() < 0.15:
        patch = add_photometric_distortion(patch)

    if random.random() < 0.85:
        patch = add_central_ui_simulation(patch)

    mask = np.zeros((CONFIG["OUTPUT_SIZE"], CONFIG["OUTPUT_SIZE"]), dtype=np.uint8)
    cv2.circle(
        mask,
        (CONFIG["OUTPUT_SIZE"] // 2, CONFIG["OUTPUT_SIZE"] // 2),
        CONFIG["MASK_DIAMETER"] // 2,
        255,
        -1,
    )
    patch = cv2.bitwise_and(patch, patch, mask=mask)
    return patch


def augment_patch_light(patch: np.ndarray) -> np.ndarray:
    """对 Tier 类执行较轻增强，避免过度扰动小样本类别。"""
    if random.random() < 0.15:
        patch = add_photometric_distortion(patch)

    if random.random() < 0.85:
        patch = add_central_ui_simulation(patch)

    mask = np.zeros((CONFIG["OUTPUT_SIZE"], CONFIG["OUTPUT_SIZE"]), dtype=np.uint8)
    cv2.circle(
        mask,
        (CONFIG["OUTPUT_SIZE"] // 2, CONFIG["OUTPUT_SIZE"] // 2),
        CONFIG["MASK_DIAMETER"] // 2,
        255,
        -1,
    )
    return cv2.bitwise_and(patch, patch, mask=mask)


# ---------------------------------------------------------------------------
# 背景合成
# ---------------------------------------------------------------------------

_bg_cache: dict = {}


def apply_background_composition(patch_bgra: np.ndarray, bg_paths: list) -> np.ndarray:
    """将带透明通道的地图图块合成到背景域上。"""
    if patch_bgra.shape[2] != 4:
        return patch_bgra

    h, w = patch_bgra.shape[:2]
    bgr = patch_bgra[..., :3].astype(np.float32)
    alpha = (patch_bgra[..., 3] / 255.0).astype(np.float32)
    alpha = np.expand_dims(alpha, axis=-1)

    rand_val = random.random()

    # 保留少量真实背景干扰，主体仍贴近游戏黑底小地图。
    if rand_val < 0.30 and bg_paths:
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

    elif rand_val < 0.80:
        bg_patch = np.zeros((h, w, 3), dtype=np.float32)

    elif rand_val < 0.90:
        gray_val = random.randint(200, 255)
        bg_patch = np.full((h, w, 3), gray_val, dtype=np.float32)

    else:
        small_noise = np.random.randint(0, 256, (16, 16, 3), dtype=np.uint8)
        bg_patch = cv2.resize(small_noise, (w, h), interpolation=cv2.INTER_NEAREST).astype(
            np.float32
        )

    bg_blend = random.uniform(*CONFIG["BACKGROUND_BLEND_RANGE"])
    result = bgr * alpha + bg_patch * (1.0 - alpha) * bg_blend
    return result.astype(np.uint8)


# ---------------------------------------------------------------------------
# 样本生成
# ---------------------------------------------------------------------------


def process_patch(
    img: np.ndarray,
    cx: int,
    cy: int,
    angle: float,
    safe_size: int,
    bg_paths: list,
    light_aug: bool = False,
) -> np.ndarray:
    """提取单个训练图块，并完成背景合成与增强。"""
    patch_bgra = extract_roi(img, cx, cy, angle, safe_size)
    patch_bgr = apply_background_composition(patch_bgra, bg_paths)

    if light_aug:
        return augment_patch_light(patch_bgr)

    return augment_patch(patch_bgr, safe_size)


def load_error_images(class_name: str, error_dir: Path) -> list:
    """加载指定类别的困难样本，并按配置过采样。"""
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
                aug = apply_random_occlusion(aug)
                error_samples.append(aug)

    return error_samples


def generate_samples(
    img: np.ndarray,
    safe_size: int,
    bg_paths: list,
    sample_region: tuple[int, int, int, int] | None = None,
    target_count: int | None = None,
    light_aug: bool = False,
    random_sampling_only: bool = False,
) -> list:
    """从单张地图切片生成 TARGET_COUNT 个增强样本。

    sample_region:
        (x, y, w, h)，坐标相对于原图（未 pad）。
        若为 None，则扫描整张原图。
    """
    target_count = target_count or CONFIG["TARGET_COUNT"]
    h, w = img.shape[:2]
    pad = safe_size // 2
    orig_h, orig_w = h - 2 * pad, w - 2 * pad

    if sample_region is None:
        region_x, region_y, region_w, region_h = 0, 0, orig_w, orig_h
    else:
        region_x, region_y, region_w, region_h = sample_region
        region_x = max(0, min(region_x, orig_w))
        region_y = max(0, min(region_y, orig_h))
        region_w = max(0, min(region_w, orig_w - region_x))
        region_h = max(0, min(region_h, orig_h - region_y))

    region_x2 = region_x + region_w
    region_y2 = region_y + region_h

    valid_centers = []
    if random_sampling_only:
        sample_attempts = max(target_count * 12, 400)
        seen_centers = set()
        for _ in range(sample_attempts):
            if region_w <= 0 or region_h <= 0:
                break

            x = random.randint(region_x, max(region_x, region_x2 - 1))
            y = random.randint(region_y, max(region_y, region_y2 - 1))
            cx, cy = x + pad, y + pad
            if (cx, cy) in seen_centers:
                continue
            seen_centers.add((cx, cy))

            if is_valid(extract_roi(img, cx, cy, 0, safe_size)):
                valid_centers.append((cx, cy))
                if len(valid_centers) >= target_count:
                    break
    else:
        for y in range(region_y, region_y2, CONFIG["STRIDE"]):
            for x in range(region_x, region_x2, CONFIG["STRIDE"]):
                cx, cy = x + pad, y + pad
                if is_valid(extract_roi(img, cx, cy, 0, safe_size)):
                    valid_centers.append((cx, cy))

    if not valid_centers:
        return []

    if sample_region is not None and len(valid_centers) < CONFIG["MIN_VALID_CENTERS_PER_TILE"]:
        return []

    samples = []
    for cx, cy in valid_centers:
        if len(samples) >= target_count:
            break
        samples.append(process_patch(img, cx, cy, 0, safe_size, bg_paths, light_aug))

    while len(samples) < target_count:
        cx, cy = random.choice(valid_centers)
        nx = max(pad, min(cx + random.randint(-5, 5), w - pad))
        ny = max(pad, min(cy + random.randint(-5, 5), h - pad))
        angle = random.uniform(-CONFIG["ANGLE_JITTER"], CONFIG["ANGLE_JITTER"])

        patch = extract_roi(img, nx, ny, angle, safe_size)
        if is_valid(patch):
            patch_bgr = apply_background_composition(patch, bg_paths)
            if light_aug:
                samples.append(augment_patch_light(patch_bgr))
            else:
                samples.append(augment_patch(patch_bgr, safe_size))

    return samples


def save_dataset(samples: list, class_name: str, file_stem: str, output_dir: Path) -> None:
    """将样本列表按 VAL_RATIO 随机划分并写入 train/val 目录。

    保证：
    - 只要该类有样本，train 至少 1 张
    - 只要该类有样本，val 至少 1 张
    这样不会出现 train/val 类集合不一致。
    """
    if not samples:
        return

    random.shuffle(samples)
    n = len(samples)

    if n == 1:
        train_samples = [samples[0]]
        val_samples = [samples[0].copy()]
    else:
        split_idx = int(n * CONFIG["VAL_RATIO"])
        split_idx = max(1, min(split_idx, n - 1))
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
    target_count_override: int | None = None,
):
    """单张源图处理入口，供 ProcessPoolExecutor 调度。"""
    safe_size = get_safe_size()

    img = load_image(file_path, safe_size)
    if img is None:
        return {
            "message": f"Failed to load {file_path}",
            "tile_mapping": {},
        }

    pad = safe_size // 2
    h, w = img.shape[:2]
    orig_h, orig_w = h - 2 * pad, w - 2 * pad

    if class_name in CONFIG["BASE_CLASS_NAMES"]:
        tiles = enumerate_base_tiles(orig_h, orig_w, class_name)
        tile_mapping = {}
        processed_count = 0

        for tile in tiles:
            tile_class = tile["class_name"]
            region = (tile["x"], tile["y"], tile["w"], tile["h"])
            samples = generate_samples(img, safe_size, bg_paths, sample_region=region)

            if error_dir and (error_dir / tile_class).exists():
                error_samples = load_error_images(tile_class, error_dir)
                if error_samples:
                    samples.extend(error_samples)

            tile_mapping[tile_class] = {
                "base_class": tile["base_class"],
                "row": tile["row"],
                "col": tile["col"],
                "x": tile["x"],
                "y": tile["y"],
                "w": tile["w"],
                "h": tile["h"],
                "infer_margin": CONFIG["TILE_INFER_MARGIN"],
            }

            if not samples:
                continue

            save_dataset(samples, tile_class, tile_class, output_dir)
            processed_count += 1

        return {
            "message": f"Completed tiled base {class_name} -> {processed_count}/{len(tiles)} tile classes with samples",
            "tile_mapping": tile_mapping,
        }

    is_none = class_name == "None"
    is_tier = "Tier" in class_name
    target_count = (
        target_count_override
        if target_count_override is not None
        else (CONFIG["TIER_TARGET_COUNT"] if is_tier else CONFIG["TARGET_COUNT"])
    )
    samples = generate_samples(
        img,
        safe_size,
        bg_paths,
        target_count=target_count,
        light_aug=is_tier,
        random_sampling_only=is_none,
    )

    # 允许所有类（包含 None）读取对应的 error_images
    if error_dir and (error_dir / class_name).exists():
        error_samples = load_error_images(class_name, error_dir)
        if error_samples:
            samples.extend(error_samples)

    save_dataset(samples, class_name, file_path.stem, output_dir)
    return {
        "message": f"Completed {class_name} ({file_path.name})",
        "tile_mapping": {},
    }


# ---------------------------------------------------------------------------
# 主流水线
# ---------------------------------------------------------------------------


class DataPreprocessor:
    """数据集生成主控类，负责目录扫描、任务调度和结果汇总。"""

    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        error_dir: str,
        bg_dir: str,
        max_workers: int | None = None,
    ):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.error_dir = Path(error_dir) if error_dir else None
        self.max_workers = max_workers

        self.bg_paths: list[Path] = []
        bg_dir_path = Path(bg_dir) if bg_dir else None
        if bg_dir_path and bg_dir_path.exists():
            for ext in ["*.[pP][nN][gG]", "*.[jJ][pP][gG]"]:
                self.bg_paths.extend(list(bg_dir_path.glob(ext)))
        logger.info(f"Discovered {len(self.bg_paths)} background images.")

    def run(self) -> None:
        """执行完整预处理流水线。"""
        if self.output_dir.exists():
            shutil.rmtree(self.output_dir)

        (self.output_dir / "train").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "val").mkdir(parents=True, exist_ok=True)

        tasks: list[tuple[Path, str, int | None]] = []
        none_files: list[Path] = []

        for ext in ["*.[pP][nN][gG]", "*.[jJ][pP][gG]"]:
            for file_path in self.input_dir.glob(ext):
                tasks.append((file_path, file_path.stem, None))

        for child in self.input_dir.iterdir():
            if not child.is_dir():
                continue
            class_name = child.name
            if class_name.startswith(".") or class_name == "__pycache__":
                continue

            logger.info(f"Queuing class directory: {class_name}")
            for ext in ["*.[pP][nN][gG]", "*.[jJ][pP][gG]"]:
                for file_path in child.glob(ext):
                    if class_name == "None":
                        none_files.append(file_path)
                    else:
                        tasks.append((file_path, class_name, None))

        if none_files:
            none_cap = CONFIG["NONE_CLASS_TOTAL_CAP"]
            min_per_file = CONFIG["NONE_PER_IMAGE_MIN"]
            max_per_file = CONFIG["NONE_PER_IMAGE_MAX"]
            per_file = max(min_per_file, min(max_per_file, none_cap // len(none_files)))
            remainder = none_cap % len(none_files)
            logger.info(
                f"Applying None class cap: {none_cap} total samples across {len(none_files)} files."
            )
            for idx, file_path in enumerate(sorted(none_files)):
                extra = 1 if idx < remainder and per_file < max_per_file else 0
                target_count = min(max_per_file, per_file + extra)
                if target_count > 0:
                    tasks.append((file_path, "None", target_count))

        max_workers = self.max_workers or os.cpu_count() or 4
        logger.info(
            f"Starting parallel processing with {max_workers} workers for {len(tasks)} tasks..."
        )

        merged_tile_mapping = {}

        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {
                executor.submit(
                    process_image_task,
                    file_path,
                    class_name,
                    self.output_dir,
                    self.error_dir,
                    self.bg_paths,
                    target_count_override,
                ): file_path
                for file_path, class_name, target_count_override in tasks
            }

            for future in concurrent.futures.as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    result = future.result()
                    logger.info(result["message"])
                    if result["tile_mapping"]:
                        merged_tile_mapping.update(result["tile_mapping"])
                except Exception as e:
                    logger.error(f"Error processing {file_path}: {e}")

        if merged_tile_mapping:
            save_tile_mapping(merged_tile_mapping, self.output_dir)
            logger.info(f"Saved tile mapping with {len(merged_tile_mapping)} entries.")

        logger.info("Preprocessing completed successfully.")


# ---------------------------------------------------------------------------
# CLI 入口
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """解析命令行覆盖项，并与默认配置合并。"""
    parser = argparse.ArgumentParser(description="Dataset Preprocessing Pipeline")
    parser.add_argument(
        "--input",
        default=argparse.SUPPRESS,
        help=f"Input directory containing source map images (default: {DEFAULT_OPTIONS['input']})",
    )
    parser.add_argument(
        "--output",
        default=argparse.SUPPRESS,
        help=f"Output directory for the generated dataset (default: {DEFAULT_OPTIONS['output']})",
    )
    parser.add_argument(
        "--error",
        default=argparse.SUPPRESS,
        help=f"Directory containing hard negative samples (default: {DEFAULT_OPTIONS['error']})",
    )
    parser.add_argument(
        "--bg",
        default=argparse.SUPPRESS,
        help=f"Directory containing background images (default: {DEFAULT_OPTIONS['bg']})",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=argparse.SUPPRESS,
        help="Parallel worker count override (default: os.cpu_count())",
    )
    parser.add_argument(
        "--target-count",
        dest="target_count",
        type=int,
        default=argparse.SUPPRESS,
        help=f"普通类别目标样本数 (default: {CONFIG['TARGET_COUNT']})",
    )
    parser.add_argument(
        "--tier-target-count",
        dest="tier_target_count",
        type=int,
        default=argparse.SUPPRESS,
        help=f"Tier 类目标样本数 (default: {CONFIG['TIER_TARGET_COUNT']})",
    )
    parser.add_argument(
        "--none-total-cap",
        dest="none_total_cap",
        type=int,
        default=argparse.SUPPRESS,
        help=f"None 类总样本上限 (default: {CONFIG['NONE_CLASS_TOTAL_CAP']})",
    )
    parser.add_argument(
        "--none-per-image-min",
        dest="none_per_image_min",
        type=int,
        default=argparse.SUPPRESS,
        help=f"单张 None 图最小采样数 (default: {CONFIG['NONE_PER_IMAGE_MIN']})",
    )
    parser.add_argument(
        "--none-per-image-max",
        dest="none_per_image_max",
        type=int,
        default=argparse.SUPPRESS,
        help=f"单张 None 图最大采样数 (default: {CONFIG['NONE_PER_IMAGE_MAX']})",
    )
    parser.add_argument(
        "--val-ratio",
        dest="val_ratio",
        type=float,
        default=argparse.SUPPRESS,
        help=f"验证集比例 (default: {CONFIG['VAL_RATIO']})",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=argparse.SUPPRESS,
        help=f"滑窗扫描步长 (default: {CONFIG['STRIDE']})",
    )
    parser.add_argument(
        "--error-oversample",
        dest="error_oversample",
        type=int,
        default=argparse.SUPPRESS,
        help=f"困难样本过采样倍数 (default: {CONFIG['ERROR_OVERSAMPLE']})",
    )

    parsed = vars(parser.parse_args())
    options = DEFAULT_OPTIONS.copy()

    for option_name, config_name in CONFIG_OVERRIDES.items():
        if option_name in parsed:
            CONFIG[config_name] = parsed.pop(option_name)

    options.update(parsed)
    return argparse.Namespace(**options)


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()

    args = parse_args()
    DataPreprocessor(
        args.input,
        args.output,
        args.error,
        args.bg,
        args.workers,
    ).run()
