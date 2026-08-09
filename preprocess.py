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

    error_images/<class_name>/      困难样本目录；仅并入训练集，不参与随机验证集
    validation_images/<class_name>/ 已按推理规格裁好的固定验证样本
    bg_images/                      背景域随机化图片目录（可选）
    map_export.json                 完整 Tier→Base 导出契约（必须由导出工具生成）
    dataset/                        输出数据集根目录，运行前自动清空

用法:
    python preprocess.py [--input <dir>] [--output <dir>]
                         [--error <dir>] [--bg <dir>]
                         [--map-export <json>] [--target-count <int>]
                         [--workers <int>]
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import math
import os
import random
import shutil
from enum import IntEnum
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
    "SCALE_JITTER_RATIO": 0.25,         # 训练专用尺度扰动样本比例
    "SCALE_JITTER_MIN": 0.90,
    "SCALE_JITTER_MAX": 1.10,
    "TIER_CENTER_DILATION": 8,          # Tier 中心只在地图结构附近采样
    "STD_THRESHOLD": 5.0,              # 保留兼容字段，实际有效性使用 MIN_VALID_STD
    "OCCLUSION_COUNT": 0,              # 保留兼容字段
    "OCCLUSION_SIZE": 0,               # 保留兼容字段
    "ERROR_OVERSAMPLE": 5,             # 困难样本过采样倍数
    "ERROR_MIN_RATIO": 0.05,           # 困难样本至少占该类生成样本的比例
    "BACKGROUND_BLEND_RANGE": (0.9, 1.0),
    "BASE_CLASS_NAMES": {"Map01Base", "Map02Base"},
    "TILE_SIZE": 160,
    "TILE_STRIDE": 160,
    "TILE_INFER_MARGIN": 64,
    "UI_ICON_SCALE": 0.08,
    "UI_ICON_SCALE_JITTER": 0.10,
    "UI_ICON_MIN_SIDE": 8,
    "UI_ICON_OUTLINE": 1,
    "UI_BLUE_PROB": 0.30,
    "UI_LANDMARK_PROB": 0.40,
    "UI_ICON_MIN_COUNT": 2,
    "UI_ICON_MAX_COUNT": 6,
    "UI_ZONE_EXTRA_RATIO": 0.50,
    "UI_ZONE_YELLOW_PROB": 0.75,
    "UI_ZONE_ALPHA_MIN": 0.20,
    "UI_ZONE_ALPHA_MAX": 0.34,
    "UI_ZONE_MIN_RADIUS": 28,
    "UI_ZONE_MAX_RADIUS": 40,
    "UI_ZONE_EDGE_LEAK_PROB": 0.35,
    "EXTREME_UI_PROB": 0.06,
    "EXTREME_UI_PACK_MIN": 1,
    "EXTREME_UI_PACK_MAX": 2,
    "EXTREME_UI_ICONS_MIN": 8,
    "EXTREME_UI_ICONS_MAX": 16,
    "EXTREME_UI_RADIUS_MIN": 10,
    "EXTREME_UI_RADIUS_MAX": 22,
    "EXTREME_UI_CHAIN_PROB": 0.5,
    "EXTREME_UI_EDGE_BIAS_PROB": 0.35,
    "ULTRA_UI_PROB": 0.01,
    "ULTRA_UI_PACK_MIN": 3,
    "ULTRA_UI_PACK_MAX": 4,
    "ULTRA_UI_ICONS_MIN": 18,
    "ULTRA_UI_ICONS_MAX": 28,
    "ULTRA_UI_RADIUS_MIN": 16,
    "ULTRA_UI_RADIUS_MAX": 32,
    "ULTRA_UI_CHAIN_PROB": 0.85,
    "TIER_CENTER_UI_EXTRA_RATIO": 0.20,
    "TIER_CENTER_UI_ICONS_MIN": 5,
    "TIER_CENTER_UI_ICONS_MAX": 8,
    "TIER_CENTER_UI_RADIUS": 18,
    "TIER_CENTER_UI_SCALE_MAX": 1.50,
    "TIER_CENTER_UI_ZONE_RATIO": 0.65,
    "UI_POINTER_PROB": 1.0,
    "MIN_MAP_CIRCLE_COVERAGE": 0.10,
    "MIN_MAP_CENTER_COVERAGE": 0.035,
    "MIN_ALPHA_CIRCLE_COVERAGE": 0.12,
    "TIER_MIN_MAP_CIRCLE_COVERAGE": 0.02,
    "TIER_MIN_ALPHA_CIRCLE_COVERAGE": 0.08,
    "TIER_PARENT_INTENSITY": 0.28,
    "MIN_VALID_STD": 8.0,
    "MIN_VALID_CENTERS_PER_TILE": 24,
}


class UiClutter(IntEnum):
    NONE = 0
    EXTREME = 1
    ULTRA = 2
    TIER_CENTER = 3


class BackgroundProfile(IntEnum):
    STANDARD = 0
    TIER = 1


DEFAULT_OPTIONS = {
    "input": "source_images",
    "output": "dataset",
    "error": "error_images",
    "bg": "bg_images",
    "fixed_val": "validation_images",
    "map_export": "map_export.json",
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


MAP_EXPORT_FORMAT = "map-cls-export-v1"


def _resolve_export_file(input_dir: Path, value: str, field: str) -> Path:
    """Resolve a manifest path while keeping it inside the source directory."""
    if not isinstance(value, str) or not value:
        raise ValueError(f"map_export.json {field} must be a non-empty relative path")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"map_export.json {field} must be relative: {value}")

    # Accept both the portable form ``Map02Base.png`` and the descriptive
    # ``source_images/Map02Base.png`` form from exported manifests.
    if path.parts and path.parts[0] == input_dir.name:
        path = Path(*path.parts[1:])
    resolved_root = input_dir.resolve()
    resolved = (input_dir / path).resolve()
    if resolved != resolved_root and resolved_root not in resolved.parents:
        raise ValueError(f"map_export.json {field} escapes the source directory: {value}")
    return resolved


def _manifest_size(value, field: str) -> tuple[int, int]:
    if not isinstance(value, list) or len(value) != 2 or any(
        isinstance(item, bool) or not isinstance(item, int) or item <= 0
        for item in value
    ):
        raise ValueError(f"map_export.json {field} must be [positive_width, positive_height]")
    return int(value[0]), int(value[1])


def _manifest_affine(value, field: str) -> tuple[float, float, float, float]:
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError(f"map_export.json {field} must contain four numbers")
    affine = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in affine) or affine[0] <= 0 or affine[2] <= 0:
        raise ValueError(f"map_export.json {field} contains an invalid affine")
    return affine


def load_map_export_manifest(path: Path, input_dir: Path) -> dict[str, dict]:
    """Load the portable Tier-to-parent contract used by CLS.

    The manifest contains only resolved facts (filenames, dimensions, and a
    diagonal affine).  No exporter module is imported here, so the training
    repository remains independent of the producer.
    """
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ValueError(f"Cannot read {path}: {error}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON in {path}: {error}") from error

    actual_format = payload.get("format") if isinstance(payload, dict) else None
    if actual_format != MAP_EXPORT_FORMAT:
        raise ValueError(
            f"{path} has format {actual_format!r}; expected {MAP_EXPORT_FORMAT!r}"
        )
    bases = payload.get("bases")
    if not isinstance(bases, dict) or not bases:
        raise ValueError(f"{path} must contain a non-empty 'bases' object")
    for base_name, raw in sorted(bases.items()):
        if not isinstance(base_name, str) or not base_name or not isinstance(raw, dict):
            raise ValueError(f"Invalid Base entry in {path}: {base_name!r}")
        base_path = _resolve_export_file(input_dir, raw.get("file", ""), "base.file")
        base_size = _manifest_size(raw.get("size"), f"{base_name}.size")
        base = safe_imread(base_path, cv2.IMREAD_UNCHANGED)
        if base is None or tuple(base.shape[:2][::-1]) != base_size:
            actual = None if base is None else (base.shape[1], base.shape[0])
            raise ValueError(
                f"{base_name} base mismatch: manifest={base_size}, actual={actual}, "
                f"path={base_path}"
            )
    tiers = payload.get("tiers")
    if not isinstance(tiers, dict) or not tiers:
        raise ValueError(f"{path} must contain a non-empty 'tiers' object")

    specs: dict[str, dict] = {}
    for class_name, raw in sorted(tiers.items()):
        if not isinstance(class_name, str) or not class_name or not isinstance(raw, dict):
            raise ValueError(f"Invalid Tier entry in {path}: {class_name!r}")

        template_path = _resolve_export_file(input_dir, raw.get("template", ""), "template")
        parent_path = _resolve_export_file(input_dir, raw.get("parent", ""), "parent")
        template_size = _manifest_size(raw.get("template_size"), f"{class_name}.template_size")
        parent_size = _manifest_size(raw.get("parent_size"), f"{class_name}.parent_size")
        affine = _manifest_affine(raw.get("tier_to_parent"), f"{class_name}.tier_to_parent")
        mask_mode = raw.get("mask_mode", "opaque")
        if mask_mode not in {"opaque", "bright"}:
            raise ValueError(
                f"{class_name}.mask_mode must be 'opaque' or 'bright', got {mask_mode!r}"
            )

        template = safe_imread(template_path, cv2.IMREAD_UNCHANGED)
        parent = safe_imread(parent_path, cv2.IMREAD_UNCHANGED)
        if template is None or tuple(template.shape[:2][::-1]) != template_size:
            actual = None if template is None else (template.shape[1], template.shape[0])
            raise ValueError(
                f"{class_name} template mismatch: manifest={template_size}, actual={actual}, "
                f"path={template_path}"
            )
        if parent is None or tuple(parent.shape[:2][::-1]) != parent_size:
            actual = None if parent is None else (parent.shape[1], parent.shape[0])
            raise ValueError(
                f"{class_name} parent mismatch: manifest={parent_size}, actual={actual}, "
                f"path={parent_path}"
            )

        specs[class_name] = {
            "template_path": str(template_path),
            "parent_path": str(parent_path),
            "template_size": template_size,
            "parent_size": parent_size,
            "affine": affine,
            "mask_mode": mask_mode,
        }
    return specs


def load_curated_sample(path: Path) -> np.ndarray:
    """读取人工样本并严格校验其已符合线上推理尺寸。"""
    image = safe_imread(path)
    expected_shape = (CONFIG["OUTPUT_SIZE"], CONFIG["OUTPUT_SIZE"])
    if image is None or image.shape[:2] != expected_shape:
        raise ValueError(
            f"Curated sample must be {expected_shape[0]}x{expected_shape[1]}: {path}"
        )
    return image


# ---------------------------------------------------------------------------
# UI 图标处理工具
# ---------------------------------------------------------------------------

_ui_icon_cache = None
UI_BLUE_BGR = (255, 209, 25)


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

    processed = {"normal": {}, "landmarks": [], "pointer": None}

    for name, img in raw_icons.items():
        if name.lower() == "pointer.png":
            processed["pointer"] = sanitize_rgba_alpha(img, alpha_floor=8)
        elif name.lower().startswith("landmark_"):
            processed["landmarks"].append(sanitize_rgba_alpha(img, alpha_floor=8))
        else:
            processed["normal"][name] = sanitize_rgba_alpha(img, alpha_floor=8)

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
    opacity: float = 0.45,
) -> np.ndarray:
    alpha = icon_rgba[..., 3]
    kernel = np.ones((thickness * 2 + 1, thickness * 2 + 1), np.uint8)
    dilated = cv2.dilate(alpha, kernel, iterations=1)
    outline_alpha = np.clip(
        (dilated.astype(np.float32) - alpha.astype(np.float32)) * opacity,
        0,
        255,
    ).astype(np.uint8)

    outline_layer = np.zeros_like(icon_rgba)
    outline_layer[..., 3] = outline_alpha

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


def tint_icon_blue_rgba(icon_rgba: np.ndarray, bgr_color=UI_BLUE_BGR) -> np.ndarray:
    """按原图明度着色，保留真实图标内部的黑色结构。"""
    result = icon_rgba.copy()
    luminance = cv2.cvtColor(result[..., :3], cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    color = np.asarray(bgr_color, dtype=np.float32)
    result[..., :3] = np.clip(luminance[..., None] * color, 0, 255).astype(np.uint8)
    result[result[..., 3] == 0, :3] = 0
    return result


def resize_rgba(icon_rgba: np.ndarray, scale: float, min_side: int | None = None) -> np.ndarray:
    h, w = icon_rgba.shape[:2]
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    if min_side is not None and min(new_w, new_h) < min_side:
        factor = min_side / max(1, min(new_w, new_h))
        new_w = int(round(new_w * factor))
        new_h = int(round(new_h * factor))

    interpolation = cv2.INTER_AREA if scale <= 1.0 else cv2.INTER_LINEAR
    alpha = icon_rgba[..., 3].astype(np.float32) / 255.0
    premultiplied = icon_rgba[..., :3].astype(np.float32) * alpha[..., None]
    resized_alpha = cv2.resize(alpha, (new_w, new_h), interpolation=interpolation)
    resized_rgb = cv2.resize(premultiplied, (new_w, new_h), interpolation=interpolation)

    result = np.zeros((new_h, new_w, 4), dtype=np.uint8)
    visible = resized_alpha > 1e-6
    result[visible, :3] = np.clip(
        resized_rgb[visible] / resized_alpha[visible, None],
        0,
        255,
    ).astype(np.uint8)
    result[..., 3] = np.clip(resized_alpha * 255, 0, 255).astype(np.uint8)
    return result


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


def _sample_normal_ui_icon(
    normal_icons: dict,
    icon_names: list[str],
    name: str | None = None,
    scale_multiplier: float = 1.0,
) -> np.ndarray:
    if name is None:
        name = random.choice(icon_names)

    jitter = CONFIG["UI_ICON_SCALE_JITTER"]
    scale = (
        CONFIG["UI_ICON_SCALE"]
        * scale_multiplier
        * random.uniform(1.0 - jitter, 1.0 + jitter)
    )
    icon = resize_rgba(
        normal_icons[name],
        scale,
        min_side=CONFIG["UI_ICON_MIN_SIDE"],
    )
    if random.random() < CONFIG["UI_BLUE_PROB"]:
        icon = tint_icon_blue_rgba(icon)

    outline = CONFIG["UI_ICON_OUTLINE"]
    icon = cv2.copyMakeBorder(
        icon,
        outline,
        outline,
        outline,
        outline,
        cv2.BORDER_CONSTANT,
        value=(0, 0, 0, 0),
    )
    return add_black_outline_rgba(icon, thickness=outline)


def draw_one_normal_icon(result: np.ndarray, icon_rgba: np.ndarray, x: int, y: int) -> np.ndarray:
    return overlay_rgba_on_bgr(result, icon_rgba, x, y)


def sample_minimap_center(h: int, w: int, edge_bias: bool = False) -> tuple[int, int]:
    """在真实小地图圆内采样图标中心，少量覆盖圆周裁切场景。"""
    max_radius = min(CONFIG["MASK_DIAMETER"] / 2 - 3, h / 2 - 3, w / 2 - 3)
    if edge_bias:
        radius = random.uniform(max_radius * 0.72, max_radius)
    else:
        radius = math.sqrt(random.random()) * max_radius
    angle = random.uniform(0, math.tau)
    return (
        round(w / 2 + math.cos(angle) * radius),
        round(h / 2 + math.sin(angle) * radius),
    )


def sample_extreme_anchor(h: int, w: int) -> tuple[int, int]:
    """采样极端 UI 干扰的锚点，同时覆盖中心簇与圆周簇。"""
    edge_bias = random.random() < CONFIG["EXTREME_UI_EDGE_BIAS_PROB"]
    return sample_minimap_center(h, w, edge_bias=edge_bias)


def add_extreme_icon_clutter(
    result: np.ndarray,
    normal_icons: dict,
    icon_names: list[str],
    ui_clutter: UiClutter,
) -> np.ndarray:
    """在局部区域叠加高密度图标，模拟强遮挡场景。"""
    if not icon_names:
        return result

    h, w = result.shape[:2]
    out = result.copy()
    ultra = ui_clutter == UiClutter.ULTRA
    prefix = "ULTRA_UI" if ultra else "EXTREME_UI"
    pack_count = random.randint(
        CONFIG[f"{prefix}_PACK_MIN"],
        CONFIG[f"{prefix}_PACK_MAX"],
    )
    total_icon_count = random.randint(
        CONFIG[f"{prefix}_ICONS_MIN"],
        CONFIG[f"{prefix}_ICONS_MAX"],
    )
    icons_per_pack = [total_icon_count // pack_count] * pack_count
    for index in range(total_icon_count % pack_count):
        icons_per_pack[index] += 1
    random.shuffle(icons_per_pack)

    for icon_count in icons_per_pack:
        cx, cy = sample_extreme_anchor(h, w)
        radius = random.randint(
            CONFIG[f"{prefix}_RADIUS_MIN"],
            CONFIG[f"{prefix}_RADIUS_MAX"],
        )
        if random.random() < CONFIG[f"{prefix}_CHAIN_PROB"]:
            angle = random.uniform(0, 2 * np.pi)
            step = random.randint(6, 9)

            for i in range(icon_count):
                t = (i - (icon_count - 1) / 2.0) * step
                normal_offset = random.gauss(0, 2.0)
                dx = int(round(np.cos(angle) * t - np.sin(angle) * normal_offset))
                dy = int(round(np.sin(angle) * t + np.cos(angle) * normal_offset))

                icon = _sample_normal_ui_icon(normal_icons, icon_names)
                ih, iw = icon.shape[:2]
                x = cx + dx - iw // 2
                y = cy + dy - ih // 2

                out = draw_one_normal_icon(out, icon, x, y)
        else:
            for _ in range(icon_count):
                dx = int(round(random.gauss(0, radius * 0.45)))
                dy = int(round(random.gauss(0, radius * 0.45)))

                icon = _sample_normal_ui_icon(normal_icons, icon_names)
                ih, iw = icon.shape[:2]
                x = cx + dx - iw // 2
                y = cy + dy - ih // 2

                out = draw_one_normal_icon(out, icon, x, y)

    return out


def add_tier_center_icon_cluster(
    result: np.ndarray,
    normal_icons: dict,
    icon_names: list[str],
    landmarks: list[np.ndarray],
) -> np.ndarray:
    """叠加靠近玩家指针的大号图标簇，覆盖真实 Tier 的中心遮挡。"""
    if not icon_names and not landmarks:
        return result

    count = random.randint(
        CONFIG["TIER_CENTER_UI_ICONS_MIN"],
        CONFIG["TIER_CENTER_UI_ICONS_MAX"],
    )
    icons = []
    if landmarks:
        icons.append(resize_rgba(random.choice(landmarks), random.uniform(0.95, 1.15)))

    icons.extend(
        _sample_normal_ui_icon(
            normal_icons,
            icon_names,
            scale_multiplier=random.uniform(
                1.0,
                CONFIG["TIER_CENTER_UI_SCALE_MAX"],
            ),
        )
        for _ in range(count - len(icons))
        if icon_names
    )
    random.shuffle(icons)

    h, w = result.shape[:2]
    radius = CONFIG["TIER_CENTER_UI_RADIUS"]
    cx = w // 2 + random.randint(-radius // 2, radius // 2)
    cy = h // 2 + random.randint(-radius // 2, radius // 2)
    out = result.copy()
    for icon in icons:
        ih, iw = icon.shape[:2]
        x = cx + round(random.gauss(0, radius * 0.45)) - iw // 2
        y = cy + round(random.gauss(0, radius * 0.45)) - ih // 2
        out = overlay_rgba_on_bgr(out, icon, x, y)

    return out


ZONE_YELLOW_BGR = (28, 168, 185)
ZONE_BLUE_BGR = (235, 205, 135)


def draw_zone_overlay(img: np.ndarray, fill_color: tuple[int, int, int]) -> np.ndarray:
    """绘制与游戏样式接近的黄色或浅蓝色任务范围圈。"""
    h, w = img.shape[:2]
    radius = random.randint(
        CONFIG["UI_ZONE_MIN_RADIUS"],
        CONFIG["UI_ZONE_MAX_RADIUS"],
    )
    map_radius = CONFIG["MASK_DIAMETER"] / 2
    center_angle = random.uniform(0, math.tau)
    center_distance = math.sqrt(random.random()) * (map_radius - 1)
    if random.random() < CONFIG["UI_ZONE_EDGE_LEAK_PROB"]:
        center_distance = random.uniform(map_radius - 8, map_radius - 1)
    cx = round(w / 2 + math.cos(center_angle) * center_distance)
    cy = round(h / 2 + math.sin(center_angle) * center_distance)

    alpha = random.uniform(
        CONFIG["UI_ZONE_ALPHA_MIN"],
        CONFIG["UI_ZONE_ALPHA_MAX"],
    )

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (cx, cy), radius, 255, -1, lineType=cv2.LINE_AA)
    mask = cv2.GaussianBlur(mask, (0, 0), random.uniform(0.6, 1.2))
    blend = mask.astype(np.float32)[..., None] * (alpha / 255.0)
    color = np.asarray(fill_color, dtype=np.float32)
    result = img.astype(np.float32) * (1.0 - blend) + color * blend
    return np.clip(result, 0, 255).astype(np.uint8)


def draw_random_ui_lines(img: np.ndarray) -> np.ndarray:
    """绘制轻量路线线条，模拟小地图上的路径 UI。"""
    h, w = img.shape[:2]
    overlay = img.copy()
    route_specs = (
        ((48, 198, 215), 4),
        ((225, 230, 230), 4),
    )

    for color, max_points in route_specs:
        if random.random() >= 0.7:
            continue
        points = [
            sample_minimap_center(h, w)
            for _ in range(random.randint(2, max_points))
        ]
        angle = random.uniform(0, math.tau)
        axis = (math.cos(angle), math.sin(angle))
        points.sort(key=lambda point: point[0] * axis[0] + point[1] * axis[1])
        cv2.polylines(
            overlay,
            [np.asarray(points, dtype=np.int32)],
            False,
            color,
            1,
            lineType=cv2.LINE_AA,
        )

    return cv2.addWeighted(overlay, 0.78, img, 0.22, 0)


def get_ui_icon_assets(icon_dir: str = "icon") -> dict:
    """返回当前进程缓存的 UI 图标资源。"""
    global _ui_icon_cache
    if _ui_icon_cache is None:
        _ui_icon_cache = load_ui_icons(icon_dir)
    return _ui_icon_cache


def add_random_map_icons(
    img: np.ndarray,
    icon_dir: str = "icon",
    ui_clutter: UiClutter = UiClutter.NONE,
) -> np.ndarray:
    """叠加可选地图图标，不包含始终存在的玩家指针。"""
    ui_icons = get_ui_icon_assets(icon_dir)

    if not ui_icons:
        return img

    h, w = img.shape[:2]
    result = img.copy()

    normal_icons = ui_icons["normal"]
    icon_names = list(normal_icons.keys())
    landmarks = ui_icons["landmarks"]
    if ui_clutter == UiClutter.TIER_CENTER:
        return add_tier_center_icon_cluster(
            result,
            normal_icons,
            icon_names,
            landmarks,
        )

    if icon_names:
        use_count = random.randint(CONFIG["UI_ICON_MIN_COUNT"], CONFIG["UI_ICON_MAX_COUNT"])

        for _ in range(use_count):
            icon = _sample_normal_ui_icon(normal_icons, icon_names)

            ih, iw = icon.shape[:2]
            cx, cy = sample_minimap_center(h, w)
            x = cx - iw // 2
            y = cy - ih // 2

            result = draw_one_normal_icon(result, icon, x, y)

        if ui_clutter != UiClutter.NONE:
            result = add_extreme_icon_clutter(
                result,
                normal_icons,
                icon_names,
                ui_clutter,
            )

    if landmarks and random.random() < CONFIG["UI_LANDMARK_PROB"]:
        landmark = resize_rgba(random.choice(landmarks), random.uniform(0.9, 1.1))
        ih, iw = landmark.shape[:2]
        cx, cy = sample_minimap_center(h, w, edge_bias=random.random() < 0.5)
        result = overlay_rgba_on_bgr(result, landmark, cx - iw // 2, cy - ih // 2)

    return result


def add_player_pointer(img: np.ndarray, icon_dir: str = "icon") -> np.ndarray:
    """将玩家指针旋转后固定叠加在小地图中心。"""
    ui_icons = get_ui_icon_assets(icon_dir)
    pointer_icon = ui_icons.get("pointer") if ui_icons else None
    if pointer_icon is None:
        raise FileNotFoundError(f"Required player pointer is missing: {icon_dir}/pointer.png")

    angle = random.uniform(0, 360)
    ih, iw = pointer_icon.shape[:2]
    center = (iw // 2, ih // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_pointer = cv2.warpAffine(
        pointer_icon,
        matrix,
        (iw, ih),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )

    h, w = img.shape[:2]
    x = (w - iw) // 2
    y = (h - ih) // 2
    return overlay_rgba_on_bgr(img.copy(), rotated_pointer, x, y)


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

    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
    elif img.ndim == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)

    pad = safe_size // 2
    return cv2.copyMakeBorder(
        img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=[0, 0, 0, 0]
    )


def premultiply_to_bgr(img: np.ndarray) -> np.ndarray:
    """Convert an RGB/RGBA image to BGR, honoring transparent pixels."""
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 3:
        return img.copy()
    alpha = img[..., 3].astype(np.float32)[..., None] / 255.0
    return (img[..., :3].astype(np.float32) * alpha).astype(np.uint8)


def build_tier_parent_context(
    template_img: np.ndarray,
    tier_spec: dict,
    safe_size: int,
) -> dict:
    """Project the complete parent Base map into the Tier template frame."""
    parent_path = Path(tier_spec["parent_path"])
    parent = safe_imread(parent_path, cv2.IMREAD_UNCHANGED)
    if parent is None:
        raise ValueError(f"Failed to load Tier parent map: {parent_path}")

    expected_size = tier_spec["parent_size"]
    actual_size = (parent.shape[1], parent.shape[0])
    if actual_size != expected_size:
        raise ValueError(
            f"Tier parent mismatch: manifest={expected_size}, actual={actual_size}, "
            f"path={parent_path}"
        )

    parent_bgr = premultiply_to_bgr(parent)
    height, width = template_img.shape[:2]
    pad = safe_size // 2
    x, y = np.meshgrid(
        np.arange(width, dtype=np.float32) - pad,
        np.arange(height, dtype=np.float32) - pad,
    )
    sx, tx, sy, ty = tier_spec["affine"]
    parent_aligned = cv2.remap(
        parent_bgr,
        sx * x + tx,
        sy * y + ty,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    return {
        "parent_aligned": parent_aligned,
        "mask_mode": tier_spec["mask_mode"],
    }


def compose_tier_context_patch(
    tier_patch_bgra: np.ndarray,
    parent_patch_bgr: np.ndarray,
    mask_mode: str = "opaque",
) -> np.ndarray:
    """Overlay the Tier foreground on the complete parent-map context."""
    tier_bgr = premultiply_to_bgr(tier_patch_bgra).astype(np.float32)
    alpha = tier_patch_bgra[..., 3].astype(np.float32) / 255.0
    gray = cv2.cvtColor(tier_bgr.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    if mask_mode == "bright":
        foreground = (alpha > 10 / 255.0) & (gray > 18)
    else:
        opaque = alpha > 10 / 255.0
        dark = (opaque & (gray <= 12)).astype(np.uint8)
        _, labels = cv2.connectedComponents(dark, 8)
        ys, xs = np.nonzero(opaque)
        if not len(xs):
            foreground = np.zeros_like(opaque)
        else:
            min_y, max_y = ys.min(), ys.max()
            min_x, max_x = xs.min(), xs.max()
            border_labels = np.unique(
                np.concatenate(
                    (
                        labels[min_y, min_x : max_x + 1],
                        labels[max_y, min_x : max_x + 1],
                        labels[min_y : max_y + 1, min_x],
                        labels[min_y : max_y + 1, max_x],
                    )
                )
            )
            border_dark = np.isin(labels, border_labels) & (labels > 0)
            foreground = opaque & ~border_dark
    foreground_alpha = alpha * foreground.astype(np.float32)
    parent = parent_patch_bgr.astype(np.float32) * CONFIG["TIER_PARENT_INTENSITY"]
    result = tier_bgr * foreground_alpha[..., None] + parent * (
        1.0 - foreground_alpha[..., None]
    )
    return np.clip(result, 0, 255).astype(np.uint8)


def extract_roi(
    img,
    cx: int,
    cy: int,
    angle: float,
    safe_size: int,
    scale: float = 1.0,
) -> np.ndarray:
    """以中心点裁出 ROI，按需旋转后返回训练尺寸图块。"""
    half = safe_size // 2
    patch = img[cy - half : cy + half, cx - half : cx + half]

    if angle != 0 or scale != 1.0:
        M = cv2.getRotationMatrix2D((half, half), angle, scale)
        border_val = (0, 0, 0, 0) if patch.shape[2] == 4 else (0, 0, 0)
        patch = cv2.warpAffine(patch, M, (safe_size, safe_size), borderValue=border_val)

    start = (safe_size - CONFIG["OUTPUT_SIZE"]) // 2
    end = start + CONFIG["OUTPUT_SIZE"]
    return patch[start:end, start:end]


def is_valid(
    patch: np.ndarray,
    *,
    min_map_circle_coverage: float | None = None,
    min_alpha_circle_coverage: float | None = None,
) -> bool:
    """判断图块是否包含足够地图内容，过滤空洞和碎片区域。"""
    min_map_circle_coverage = (
        CONFIG["MIN_MAP_CIRCLE_COVERAGE"]
        if min_map_circle_coverage is None
        else min_map_circle_coverage
    )
    min_alpha_circle_coverage = (
        CONFIG["MIN_ALPHA_CIRCLE_COVERAGE"]
        if min_alpha_circle_coverage is None
        else min_alpha_circle_coverage
    )
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
        if alpha_circle_ratio < min_alpha_circle_coverage:
            return False
    else:
        bgr = patch.copy()
        alpha_bool = np.ones((size, size), dtype=bool)

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    map_bool = (gray > 12) & alpha_bool & circle_bool

    map_circle_ratio = np.count_nonzero(map_bool) / circle_area
    if map_circle_ratio < min_map_circle_coverage:
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


def add_central_ui_simulation(
    img: np.ndarray,
    feature_strength: float | None = None,
    ui_clutter: UiClutter = UiClutter.NONE,
) -> np.ndarray:
    """叠加可选路线和地图图标干扰。"""
    result = img.copy()
    line_passes = 2 if ui_clutter == UiClutter.ULTRA else 1
    for _ in range(line_passes):
        result = draw_random_ui_lines(result)
    result = add_random_map_icons(result, "icon", ui_clutter)
    return result


def apply_minimap_mask(img: np.ndarray) -> np.ndarray:
    """清除圆形小地图区域外的增强内容。"""
    mask = np.zeros((CONFIG["OUTPUT_SIZE"], CONFIG["OUTPUT_SIZE"]), dtype=np.uint8)
    cv2.circle(
        mask,
        (CONFIG["OUTPUT_SIZE"] // 2, CONFIG["OUTPUT_SIZE"] // 2),
        CONFIG["MASK_DIAMETER"] // 2,
        255,
        -1,
    )
    return cv2.bitwise_and(img, img, mask=mask)


def finalize_positive_map_sample(img: np.ndarray) -> np.ndarray:
    """为地图正样本补上必选玩家指针并重新收紧圆形遮罩。"""
    return apply_minimap_mask(add_player_pointer(img))


def augment_patch(
    patch: np.ndarray,
    safe_size: int,
    ui_clutter: UiClutter = UiClutter.NONE,
) -> np.ndarray:
    """对普通类别样本执行完整增强。"""
    if random.random() < 0.15:
        patch = add_photometric_distortion(patch)

    if ui_clutter != UiClutter.NONE or random.random() < 0.85:
        patch = add_central_ui_simulation(patch, ui_clutter=ui_clutter)

    return apply_minimap_mask(patch)


def augment_patch_light(
    patch: np.ndarray,
    ui_clutter: UiClutter = UiClutter.NONE,
) -> np.ndarray:
    """对 Tier 类执行较轻增强，避免过度扰动小样本类别。"""
    if random.random() < 0.15:
        patch = add_photometric_distortion(patch)

    if ui_clutter != UiClutter.NONE or random.random() < 0.85:
        patch = add_central_ui_simulation(patch, ui_clutter=ui_clutter)

    return apply_minimap_mask(patch)


def augment_zone_patch(
    patch: np.ndarray,
    fill_color: tuple[int, int, int],
    ui_clutter: UiClutter = UiClutter.NONE,
) -> np.ndarray:
    """生成一个必定包含区域圈的额外地图样本。"""
    if random.random() < 0.15:
        patch = add_photometric_distortion(patch)

    patch = draw_zone_overlay(patch, fill_color)
    if ui_clutter != UiClutter.NONE or random.random() < 0.85:
        patch = add_central_ui_simulation(patch, ui_clutter=ui_clutter)

    return apply_minimap_mask(patch)


# ---------------------------------------------------------------------------
# 背景合成
# ---------------------------------------------------------------------------

_bg_cache: dict = {}


def apply_background_composition(
    patch_bgra: np.ndarray,
    bg_paths: list,
    profile: BackgroundProfile = BackgroundProfile.STANDARD,
) -> np.ndarray:
    """将带透明通道的地图图块合成到背景域上。"""
    if patch_bgra.shape[2] != 4:
        return patch_bgra

    h, w = patch_bgra.shape[:2]
    bgr = patch_bgra[..., :3].astype(np.float32)
    alpha = (patch_bgra[..., 3] / 255.0).astype(np.float32)
    alpha = np.expand_dims(alpha, axis=-1)

    rand_val = random.random()

    # 保留少量真实背景干扰，主体仍贴近游戏黑底小地图。
    if profile != BackgroundProfile.TIER and rand_val < 0.30 and bg_paths:
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

    elif rand_val < 0.80 or profile == BackgroundProfile.TIER:
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


def compose_patch(
    img: np.ndarray,
    cx: int,
    cy: int,
    angle: float,
    safe_size: int,
    bg_paths: list,
    scale: float = 1.0,
    background_profile: BackgroundProfile = BackgroundProfile.STANDARD,
    tier_context: dict | None = None,
) -> np.ndarray:
    """提取图块并合成背景。"""
    patch_bgra = extract_roi(img, cx, cy, angle, safe_size, scale)
    if tier_context is not None:
        parent_patch = extract_roi(
            tier_context["parent_aligned"],
            cx,
            cy,
            angle,
            safe_size,
            scale,
        )
        return compose_tier_context_patch(
            patch_bgra,
            parent_patch,
            tier_context["mask_mode"],
        )
    return apply_background_composition(patch_bgra, bg_paths, background_profile)


def build_tier_center_mask(
    img: np.ndarray,
    mask_mode: str = "opaque",
) -> np.ndarray:
    """生成 Tier 地图结构及其邻近区域的合法中心掩码。"""
    alpha = img[..., 3]
    premultiplied = (
        img[..., :3].astype(np.float32) * (alpha.astype(np.float32)[..., None] / 255.0)
    ).astype(np.uint8)
    gray = cv2.cvtColor(premultiplied, cv2.COLOR_BGR2GRAY)
    gray_threshold = 18 if mask_mode == "bright" else 12
    structure = ((alpha > 10) & (gray > gray_threshold)).astype(np.uint8)
    radius = CONFIG["TIER_CENTER_DILATION"]
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (radius * 2 + 1, radius * 2 + 1),
    )
    return cv2.dilate(structure, kernel) > 0


def build_balanced_schedule(items: list, count: int) -> list:
    """按完整随机轮次均衡取样，使任意两项的出现次数最多相差一次。"""
    if not items or count <= 0:
        return []

    schedule = []
    while len(schedule) < count:
        cycle = items.copy()
        random.shuffle(cycle)
        schedule.extend(cycle[: count - len(schedule)])
    return schedule


def build_ui_clutter_schedule(sample_count: int) -> list[UiClutter]:
    """保持原密集比例，并将其中少量样本固定为超极端 UI。"""
    extreme_count = round(sample_count * 0.85 * CONFIG["EXTREME_UI_PROB"])
    ultra_count = round(sample_count * 0.85 * CONFIG["ULTRA_UI_PROB"])
    if extreme_count:
        ultra_count = min(extreme_count, max(1, ultra_count))

    schedule = (
        [UiClutter.ULTRA] * ultra_count
        + [UiClutter.EXTREME] * (extreme_count - ultra_count)
        + [UiClutter.NONE] * (sample_count - extreme_count)
    )
    random.shuffle(schedule)
    return schedule


def build_zone_ui_clutter_schedule(
    zone_plan: list[tuple[tuple[int, int], tuple[int, int, int]]],
) -> list[UiClutter]:
    """分别保证黄圈与蓝圈样本中的密集强度覆盖。"""
    schedule = [UiClutter.NONE] * len(zone_plan)
    for fill_color in (ZONE_YELLOW_BGR, ZONE_BLUE_BGR):
        indices = [
            index
            for index, (_center, color) in enumerate(zone_plan)
            if color == fill_color
        ]
        for index, ui_clutter in zip(indices, build_ui_clutter_schedule(len(indices))):
            schedule[index] = ui_clutter
    return schedule


def build_scale_jitter_schedule(sample_count: int) -> list[float]:
    """按固定配额生成对称尺度扰动，其余样本保持原尺寸。"""
    jitter_count = round(sample_count * CONFIG["SCALE_JITTER_RATIO"])
    smaller_count = (jitter_count + 1) // 2
    larger_count = jitter_count - smaller_count
    schedule = [
        random.uniform(CONFIG["SCALE_JITTER_MIN"], 1.0)
        for _ in range(smaller_count)
    ]
    schedule.extend(
        random.uniform(1.0, CONFIG["SCALE_JITTER_MAX"])
        for _ in range(larger_count)
    )
    schedule.extend([1.0] * (sample_count - jitter_count))
    random.shuffle(schedule)
    return schedule


def build_zone_plan(
    valid_centers: list[tuple[int, int]],
    zone_count: int,
) -> tuple[
    list[tuple[tuple[int, int], tuple[int, int, int]]],
    list[tuple[tuple[int, int], tuple[int, int, int]]],
]:
    """优先留存黄色位置覆盖，并返回可参与数据集切分的剩余圈计划。"""
    yellow_count = round(zone_count * CONFIG["UI_ZONE_YELLOW_PROB"])
    blue_count = zone_count - yellow_count
    yellow_plan = [
        (center, ZONE_YELLOW_BGR)
        for center in build_balanced_schedule(valid_centers, yellow_count)
    ]
    coverage_count = min(len(valid_centers), len(yellow_plan))
    train_only_plan = yellow_plan[:coverage_count]
    split_plan = yellow_plan[coverage_count:]
    split_plan.extend(
        (center, ZONE_BLUE_BGR)
        for center in build_balanced_schedule(valid_centers, blue_count)
    )
    random.shuffle(train_only_plan)
    random.shuffle(split_plan)
    return train_only_plan, split_plan


def build_tier_center_ui_plan(
    valid_centers: list[tuple[int, int]],
    target_count: int,
) -> list[tuple[tuple[int, int], tuple[int, int, int] | None]]:
    """让每个 Tier 合法中心至少包含一次原尺寸中心图标簇。"""
    count = max(
        len(valid_centers),
        round(target_count * CONFIG["TIER_CENTER_UI_EXTRA_RATIO"]),
    )
    centers = build_balanced_schedule(valid_centers, count)
    zone_count = round(count * CONFIG["TIER_CENTER_UI_ZONE_RATIO"])
    yellow_count = round(zone_count * CONFIG["UI_ZONE_YELLOW_PROB"])
    colors = (
        [ZONE_YELLOW_BGR] * yellow_count
        + [ZONE_BLUE_BGR] * (zone_count - yellow_count)
        + [None] * (count - zone_count)
    )
    random.shuffle(colors)
    return list(zip(centers, colors))


def load_error_images(
    class_name: str,
    error_dir: Path,
    generated_count: int,
) -> tuple[list, list]:
    """加载指定类别的困难样本，并生成稳定且有实际权重的训练样本。"""
    source_images = []
    error_path = error_dir / class_name

    for ext in ("*.[pP][nN][gG]", "*.[jJ][pP][gG]"):
        for file_path in sorted(error_path.glob(ext)):
            source_images.append(load_curated_sample(file_path))

    sample_count = max(
        len(source_images) * CONFIG["ERROR_OVERSAMPLE"],
        round(generated_count * CONFIG["ERROR_MIN_RATIO"]),
    )
    return [
        apply_random_occlusion(img.copy())
        for img in build_balanced_schedule(source_images, sample_count)
    ]


def add_error_training_samples(error_dir: Path | None, output_dir: Path) -> int:
    """将困难样本按类别一次性写入 train，避免跨源图重复和验证集泄漏。"""
    if error_dir is None or not error_dir.exists():
        return 0

    total = 0
    class_dirs = sorted(
        path
        for path in error_dir.iterdir()
        if path.is_dir() and not path.name.startswith((".", "_"))
    )
    for class_dir in class_dirs:
        train_dir = output_dir / "train" / class_dir.name
        val_dir = output_dir / "val" / class_dir.name
        if not train_dir.exists() or not val_dir.exists():
            logger.warning(
                f"Skipping hard samples for unknown generated class: {class_dir.name}"
            )
            continue

        generated_count = sum(
            1
            for split_dir in (train_dir, val_dir)
            for path in split_dir.iterdir()
            if path.suffix.lower() in {".png", ".jpg", ".jpeg"}
        )
        error_samples = load_error_images(
            class_dir.name,
            error_dir,
            generated_count,
        )
        for index, img in enumerate(error_samples):
            safe_imwrite(train_dir / f"hard_{index:05d}.jpg", img)

        total += len(error_samples)
        if error_samples:
            logger.info(
                f"Added {len(error_samples)} train-only hard samples for {class_dir.name}."
            )

    return total


def generate_samples(
    img: np.ndarray,
    safe_size: int,
    bg_paths: list,
    sample_region: tuple[int, int, int, int] | None = None,
    target_count: int | None = None,
    light_aug: bool = False,
    random_sampling_only: bool = False,
    tier_context: dict | None = None,
) -> tuple[list, list]:
    """生成 TARGET_COUNT 个基础样本，并为地图类追加区域圈干扰样本。

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
    tier_center_mask = (
        build_tier_center_mask(
            img[pad : pad + orig_h, pad : pad + orig_w],
            tier_context["mask_mode"] if tier_context is not None else "opaque",
        )
        if light_aug
        else None
    )
    background_profile = (
        BackgroundProfile.TIER if light_aug else BackgroundProfile.STANDARD
    )
    min_map_circle_coverage = (
        CONFIG["TIER_MIN_MAP_CIRCLE_COVERAGE"]
        if light_aug
        else CONFIG["MIN_MAP_CIRCLE_COVERAGE"]
    )
    min_alpha_circle_coverage = (
        CONFIG["TIER_MIN_ALPHA_CIRCLE_COVERAGE"]
        if light_aug
        else CONFIG["MIN_ALPHA_CIRCLE_COVERAGE"]
    )

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

            if is_valid(
                extract_roi(img, cx, cy, 0, safe_size),
                min_map_circle_coverage=min_map_circle_coverage,
                min_alpha_circle_coverage=min_alpha_circle_coverage,
            ):
                valid_centers.append((cx, cy))
                if len(valid_centers) >= target_count:
                    break
    else:
        for y in range(region_y, region_y2, CONFIG["STRIDE"]):
            for x in range(region_x, region_x2, CONFIG["STRIDE"]):
                if tier_center_mask is not None and not tier_center_mask[y, x]:
                    continue
                cx, cy = x + pad, y + pad
                if is_valid(
                    extract_roi(img, cx, cy, 0, safe_size),
                    min_map_circle_coverage=min_map_circle_coverage,
                    min_alpha_circle_coverage=min_alpha_circle_coverage,
                ):
                    valid_centers.append((cx, cy))

    if not valid_centers:
        return [], []

    if sample_region is not None and len(valid_centers) < CONFIG["MIN_VALID_CENTERS_PER_TILE"]:
        return [], []

    samples = []
    train_only_samples = []
    center_schedule = build_balanced_schedule(valid_centers, target_count)
    ui_clutter_schedule = build_ui_clutter_schedule(len(center_schedule))
    scale_schedule = (
        [1.0] * len(center_schedule)
        if random_sampling_only
        else build_scale_jitter_schedule(len(center_schedule))
    )
    if sample_region is not None:
        # Base 每个合法位置的首轮样本保持干净、原比例；其余轮次承接全部扰动配额。
        ui_clutter_schedule.sort(key=lambda level: level != UiClutter.NONE)
        scale_schedule.sort(key=lambda scale: scale != 1.0)
    min_cx, max_cx = region_x + pad, region_x2 - 1 + pad
    min_cy, max_cy = region_y + pad, region_y2 - 1 + pad

    for index, ((cx, cy), ui_clutter, scale) in enumerate(
        zip(center_schedule, ui_clutter_schedule, scale_schedule)
    ):
        nx, ny, angle = cx, cy, 0.0
        if index >= len(valid_centers):
            # Tile 标签由中心点决定，随机位移不能跨入相邻 tile。
            nx = min(max(cx + random.randint(-5, 5), min_cx), max_cx)
            ny = min(max(cy + random.randint(-5, 5), min_cy), max_cy)
            angle = random.uniform(-CONFIG["ANGLE_JITTER"], CONFIG["ANGLE_JITTER"])
            if tier_center_mask is not None and not tier_center_mask[ny - pad, nx - pad]:
                nx, ny = cx, cy

        patch = extract_roi(img, nx, ny, angle, safe_size, scale)
        if not is_valid(
            patch,
            min_map_circle_coverage=min_map_circle_coverage,
            min_alpha_circle_coverage=min_alpha_circle_coverage,
        ):
            nx, ny, angle = cx, cy, 0.0
            patch = extract_roi(img, cx, cy, 0, safe_size)
            scale = 1.0

        if tier_context is None:
            patch_bgr = apply_background_composition(
                patch,
                bg_paths,
                background_profile,
            )
        else:
            parent_patch = extract_roi(
                tier_context["parent_aligned"],
                nx,
                ny,
                angle,
                safe_size,
                scale,
            )
            patch_bgr = compose_tier_context_patch(
                patch,
                parent_patch,
                tier_context["mask_mode"],
            )
        base_anchor = sample_region is not None and index < len(valid_centers)
        if base_anchor:
            sample = apply_minimap_mask(patch_bgr)
        elif light_aug:
            sample = augment_patch_light(patch_bgr, ui_clutter)
        else:
            sample = augment_patch(patch_bgr, safe_size, ui_clutter)

        if not random_sampling_only:
            sample = finalize_positive_map_sample(sample)

        train_only = ui_clutter != UiClutter.NONE or scale != 1.0 or (
            sample_region is not None and index < len(valid_centers)
        )
        target = train_only_samples if train_only else samples
        target.append(sample)

    if light_aug:
        for (cx, cy), fill_color in build_tier_center_ui_plan(
            valid_centers,
            target_count,
        ):
            patch_bgr = compose_patch(
                img,
                cx,
                cy,
                0,
                safe_size,
                bg_paths,
                1.0,
                background_profile=background_profile,
                tier_context=tier_context,
            )
            sample = (
                augment_patch_light(patch_bgr, UiClutter.TIER_CENTER)
                if fill_color is None
                else augment_zone_patch(
                    patch_bgr,
                    fill_color,
                    UiClutter.TIER_CENTER,
                )
            )
            train_only_samples.append(finalize_positive_map_sample(sample))

    if random_sampling_only:
        return samples, train_only_samples

    zone_count = round(target_count * CONFIG["UI_ZONE_EXTRA_RATIO"])
    train_zone_plan, split_zone_plan = build_zone_plan(valid_centers, zone_count)
    if sample_region is None:
        split_zone_plan.extend(train_zone_plan)
        random.shuffle(split_zone_plan)
        train_zone_plan = []

    train_zone_ui_clutter = build_zone_ui_clutter_schedule(train_zone_plan)
    split_zone_ui_clutter = build_zone_ui_clutter_schedule(split_zone_plan)
    train_zone_scales = build_scale_jitter_schedule(len(train_zone_plan))
    split_zone_scales = build_scale_jitter_schedule(len(split_zone_plan))
    for ((cx, cy), fill_color), ui_clutter, scale in zip(
        train_zone_plan,
        train_zone_ui_clutter,
        train_zone_scales,
    ):
        patch_bgr = compose_patch(
            img,
            cx,
            cy,
            0,
            safe_size,
            bg_paths,
            scale,
            background_profile=background_profile,
            tier_context=tier_context,
        )
        sample = augment_zone_patch(patch_bgr, fill_color, ui_clutter)
        train_only_samples.append(finalize_positive_map_sample(sample))
    for ((cx, cy), fill_color), ui_clutter, scale in zip(
        split_zone_plan,
        split_zone_ui_clutter,
        split_zone_scales,
    ):
        patch_bgr = compose_patch(
            img,
            cx,
            cy,
            0,
            safe_size,
            bg_paths,
            scale,
            background_profile=background_profile,
            tier_context=tier_context,
        )
        sample = augment_zone_patch(patch_bgr, fill_color, ui_clutter)
        train_only = ui_clutter != UiClutter.NONE or scale != 1.0
        target = train_only_samples if train_only else samples
        target.append(finalize_positive_map_sample(sample))
    return samples, train_only_samples


def save_dataset(
    samples: list,
    class_name: str,
    file_stem: str,
    output_dir: Path,
    train_only_samples: list | None = None,
) -> None:
    """将样本列表按 VAL_RATIO 随机划分并写入 train/val 目录。

    train_only_samples 用于保留每个位置的普通/黄圈监督，不参与随机验证集。

    保证：
    - 只要该类有样本，train 至少 1 张
    - 只要该类有样本，val 至少 1 张
    这样不会出现 train/val 类集合不一致。
    """
    train_only_samples = train_only_samples or []
    if not samples and not train_only_samples:
        return

    random.shuffle(samples)
    total = len(samples) + len(train_only_samples)

    if total == 1:
        sample = (samples or train_only_samples)[0]
        train_samples = [sample]
        val_samples = [sample.copy()]
    elif not samples:
        train_samples = train_only_samples
        val_samples = [random.choice(train_only_samples).copy()]
    else:
        val_count = max(1, min(int(total * CONFIG["VAL_RATIO"]), len(samples)))
        if not train_only_samples:
            val_count = min(val_count, len(samples) - 1)
        val_samples = samples[:val_count]
        train_samples = samples[val_count:] + train_only_samples
        random.shuffle(train_samples)

    train_class_dir = output_dir / "train" / class_name
    val_class_dir = output_dir / "val" / class_name
    train_class_dir.mkdir(parents=True, exist_ok=True)
    val_class_dir.mkdir(parents=True, exist_ok=True)

    for i, img in enumerate(train_samples):
        safe_imwrite(train_class_dir / f"{file_stem}_{i:05d}.jpg", img)

    for i, img in enumerate(val_samples):
        safe_imwrite(val_class_dir / f"{file_stem}_{i:05d}.jpg", img)


def copy_fixed_validation_samples(fixed_val_dir: Path, output_dir: Path) -> int:
    """校验并将人工确认的预处理样本原样加入验证集。"""
    if not fixed_val_dir.exists():
        return 0

    copied = 0
    class_dirs = sorted(
        path
        for path in fixed_val_dir.iterdir()
        if path.is_dir() and not path.name.startswith(".")
    )
    for class_dir in class_dirs:
        target_dir = output_dir / "val" / class_dir.name
        if not (output_dir / "train" / class_dir.name).exists():
            raise ValueError(
                f"Fixed validation class was not generated: {class_dir.name}"
            )

        for pattern in ("*.[pP][nN][gG]", "*.[jJ][pP][gG]"):
            for source_path in sorted(class_dir.glob(pattern)):
                load_curated_sample(source_path)
                target_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_path, target_dir / f"fixed_{source_path.name}")
                copied += 1
    return copied


# ---------------------------------------------------------------------------
# 并行任务入口
# ---------------------------------------------------------------------------


def process_image_task(
    file_path: Path,
    class_name: str,
    output_dir: Path,
    bg_paths: list,
    target_count_override: int | None = None,
    tier_spec: dict | None = None,
):
    """单张源图处理入口，供 ProcessPoolExecutor 调度。"""
    safe_size = get_safe_size()

    img = load_image(file_path, safe_size)
    if img is None:
        return {
            "message": f"Failed to load {file_path}",
            "tile_mapping": {},
        }

    is_tier = "Tier" in class_name
    tier_context = None
    if is_tier:
        if tier_spec is None:
            raise ValueError(f"Missing map_export.json entry for Tier class: {class_name}")
        if Path(tier_spec["template_path"]).resolve() != file_path.resolve():
            raise ValueError(
                f"Tier manifest template does not match task: {class_name} -> "
                f"{tier_spec['template_path']} (task={file_path})"
            )
        tier_context = build_tier_parent_context(img, tier_spec, safe_size)

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
            samples, train_only_samples = generate_samples(
                img,
                safe_size,
                bg_paths,
                sample_region=region,
            )

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

            if not samples and not train_only_samples:
                continue

            save_dataset(
                samples,
                tile_class,
                tile_class,
                output_dir,
                train_only_samples,
            )
            processed_count += 1

        return {
            "message": f"Completed tiled base {class_name} -> {processed_count}/{len(tiles)} tile classes with samples",
            "tile_mapping": tile_mapping,
        }

    is_none = class_name == "None"
    target_count = (
        target_count_override
        if target_count_override is not None
        else (CONFIG["TIER_TARGET_COUNT"] if is_tier else CONFIG["TARGET_COUNT"])
    )
    samples, train_only_samples = generate_samples(
        img,
        safe_size,
        bg_paths,
        target_count=target_count,
        light_aug=is_tier,
        random_sampling_only=is_none,
        tier_context=tier_context,
    )

    save_dataset(
        samples,
        class_name,
        file_path.stem,
        output_dir,
        train_only_samples,
    )
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
        fixed_val_dir: str,
        max_workers: int | None = None,
        map_export_path: str = DEFAULT_OPTIONS["map_export"],
    ):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.error_dir = Path(error_dir) if error_dir else None
        self.fixed_val_dir = Path(fixed_val_dir)
        self.max_workers = max_workers
        self.map_export_path = Path(map_export_path)
        self.tier_specs = None
        if self.map_export_path.exists():
            self.tier_specs = load_map_export_manifest(
                self.map_export_path,
                self.input_dir,
            )

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

        tier_classes = {
            class_name
            for _file_path, class_name, _target_count in tasks
            if "Tier" in class_name
        }
        if tier_classes:
            if self.tier_specs is None:
                raise FileNotFoundError(
                    f"Tier classes require the complete export manifest: {self.map_export_path}"
                )
            manifest_classes = set(self.tier_specs)
            missing = sorted(tier_classes - manifest_classes)
            extra = sorted(manifest_classes - tier_classes)
            if missing or extra:
                details = []
                if missing:
                    details.append(f"missing entries: {', '.join(missing)}")
                if extra:
                    details.append(f"stale entries: {', '.join(extra)}")
                raise ValueError("map_export.json does not match source_images (" + "; ".join(details) + ")")

        max_workers = self.max_workers or os.cpu_count() or 4
        logger.info(
            f"Starting parallel processing with {max_workers} workers for {len(tasks)} tasks..."
        )

        merged_tile_mapping = {}
        failed_tasks = []

        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {
                executor.submit(
                    process_image_task,
                    file_path,
                    class_name,
                    self.output_dir,
                    self.bg_paths,
                    target_count_override,
                    self.tier_specs.get(class_name) if self.tier_specs else None,
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
                    failed_tasks.append((file_path, e))

        if failed_tasks:
            summary = "; ".join(f"{path}: {error}" for path, error in failed_tasks[:5])
            if len(failed_tasks) > 5:
                summary += f"; ... and {len(failed_tasks) - 5} more"
            raise RuntimeError(f"Preprocessing failed for {len(failed_tasks)} task(s): {summary}")

        if merged_tile_mapping:
            save_tile_mapping(merged_tile_mapping, self.output_dir)
            logger.info(f"Saved tile mapping with {len(merged_tile_mapping)} entries.")

        hard_sample_count = add_error_training_samples(
            self.error_dir,
            self.output_dir,
        )
        logger.info(f"Added {hard_sample_count} train-only hard samples in total.")

        fixed_val_count = copy_fixed_validation_samples(
            self.fixed_val_dir,
            self.output_dir,
        )
        logger.info(f"Added {fixed_val_count} fixed validation samples.")

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
        "--fixed-val",
        dest="fixed_val",
        default=argparse.SUPPRESS,
        help=f"Directory containing fixed validation samples (default: {DEFAULT_OPTIONS['fixed_val']})",
    )
    parser.add_argument(
        "--map-export",
        dest="map_export",
        default=argparse.SUPPRESS,
        help=f"Portable Tier export manifest (default: {DEFAULT_OPTIONS['map_export']})",
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
        args.fixed_val,
        args.workers,
        args.map_export,
    ).run()
