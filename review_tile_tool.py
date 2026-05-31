"""
review_tile_tool.py — Base tile 映射审阅图生成工具

读取 tile_mapping.json，在 Base 原图上绘制 tile 网格和 row/col 标签，用于核对
preprocess.py 生成的 Base tile 映射。默认输出到 review_grids/。

用法:
    python review_tile_tool.py [--tile-mapping <path>] [--base-dir <dir>] [--output <dir>]
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


DEFAULT_OPTIONS = {
    "tile_mapping": "tile_mapping.json",
    "base_dir": "source_images",
    "output": "review_grids",
}
BASE_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


def safe_imread(path: Path, flags=cv2.IMREAD_COLOR):
    """读取图片，兼容 Path 对象和非 ASCII 文件名。"""
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, flags)


def load_mapping(mapping_path: Path) -> dict[str, list[tuple[str, dict]]]:
    """按 base_class 分组加载 tile 映射。"""
    with open(mapping_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    by_base = defaultdict(list)
    for class_name, info in raw.items():
        base = info["base_class"]
        by_base[base].append((class_name, info))

    return {
        base_name: sorted(items, key=lambda item: (item[1].get("row", -1), item[1].get("col", -1)))
        for base_name, items in by_base.items()
    }


def draw_grid(image: np.ndarray, items, base_name: str) -> np.ndarray:
    """在 Base 原图上绘制 tile 边框和标签。"""
    canvas = image.copy()
    line_color = (0, 0, 255)
    text_color = (0, 0, 255)
    shadow = (0, 0, 0)

    for class_name, info in items:
        x = int(info["x"])
        y = int(info["y"])
        w = int(info["w"])
        h = int(info["h"])
        row = int(info.get("row", -1))
        col = int(info.get("col", -1))
        x2 = min(x + w, canvas.shape[1] - 1)
        y2 = min(y + h, canvas.shape[0] - 1)
        cv2.rectangle(canvas, (x, y), (x2, y2), line_color, 2, cv2.LINE_AA)
        label = f"r{row:02d}_c{col:02d}"
        tx = max(4, min(x + 6, canvas.shape[1] - 120))
        ty = max(24, min(y + 28, canvas.shape[0] - 8))
        cv2.putText(canvas, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.9, shadow, 5, cv2.LINE_AA)
        cv2.putText(canvas, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.9, text_color, 2, cv2.LINE_AA)

    cv2.putText(canvas, base_name, (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.1, shadow, 6, cv2.LINE_AA)
    cv2.putText(canvas, base_name, (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.1, text_color, 2, cv2.LINE_AA)
    return canvas


def save_image(path: Path, image: np.ndarray):
    """写入图片，自动创建父目录。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower() or ".png"
    ok, buf = cv2.imencode(ext, image)
    if not ok:
        raise RuntimeError(f"Failed to encode image: {path}")
    buf.tofile(str(path))


def find_base_image(base_dir: Path, base_name: str) -> Path | None:
    """查找 Base 原图文件。"""
    for ext in BASE_IMAGE_EXTENSIONS:
        path = base_dir / f"{base_name}{ext}"
        if path.exists():
            return path
    return None


def generate_review_grids(tile_mapping: str, base_dir: str, output: str) -> None:
    """从 tile 映射生成所有 Base 审阅网格图。"""
    mapping_path = Path(tile_mapping)
    base_dir_path = Path(base_dir)
    output_dir = Path(output)
    by_base = load_mapping(mapping_path)

    if not by_base:
        raise SystemExit("No tiles found in mapping.")

    for base_name, items in by_base.items():
        base_path = find_base_image(base_dir_path, base_name)
        if base_path is None:
            print(f"[WARN] Base image not found for {base_name}, skipped.")
            continue

        img = safe_imread(base_path)
        if img is None:
            print(f"[WARN] Failed to read {base_path}, skipped.")
            continue

        out = draw_grid(img, items, base_name)
        out_path = output_dir / f"{base_name}__grid.png"
        save_image(out_path, out)
        print(f"[OK] {out_path}")


def parse_args() -> argparse.Namespace:
    """解析命令行覆盖项，并与 DEFAULT_OPTIONS 合并。"""
    parser = argparse.ArgumentParser(description="Draw review grid strictly from tile_mapping.json.")
    parser.add_argument(
        "--tile-mapping",
        default=argparse.SUPPRESS,
        help=f"Path to tile_mapping.json (default: {DEFAULT_OPTIONS['tile_mapping']})",
    )
    parser.add_argument(
        "--base-dir",
        default=argparse.SUPPRESS,
        help=f"Directory containing base map images (default: {DEFAULT_OPTIONS['base_dir']})",
    )
    parser.add_argument(
        "--output",
        default=argparse.SUPPRESS,
        help=f"Output directory for generated grid images (default: {DEFAULT_OPTIONS['output']})",
    )

    options = DEFAULT_OPTIONS.copy()
    options.update(vars(parser.parse_args()))
    return argparse.Namespace(**options)


if __name__ == "__main__":
    args = parse_args()
    generate_review_grids(args.tile_mapping, args.base_dir, args.output)
