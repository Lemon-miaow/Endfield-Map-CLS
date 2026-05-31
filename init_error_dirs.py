"""
init_error_dirs.py — 困难负样本目录初始化脚本

扫描 source_images 目录中的原始地图图片，在 error_images 下创建类别目录。
Base 大图会按训练网格展开成 tile 类目录，并额外创建 _pending 目录用于先暂存
未拆分的 Base 错图。

用法:
    python init_error_dirs.py [--source <dir>] [--error <dir>]
"""

import argparse
from pathlib import Path

import cv2
import numpy as np


BASE_CLASS_NAMES = {"Map01Base", "Map02Base"}
TILE_SIZE = 320
TILE_STRIDE = 160
DEFAULT_OPTIONS = {
    "source": "source_images",
    "error": "error_images",
}


def safe_imread(path, flags=cv2.IMREAD_COLOR):
    """读取图片，兼容路径中包含非 ASCII 字符（如中文）的情况。"""
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), flags)


def build_axis_positions(length: int, tile_size: int, stride: int) -> list[int]:
    """构建覆盖整张 Base 图的 tile 起点。"""
    if length <= tile_size:
        return [0]

    positions = list(range(0, length - tile_size + 1, stride))
    last = length - tile_size
    if positions[-1] != last:
        positions.append(last)
    return positions


def enumerate_base_tile_names(img_h: int, img_w: int, class_name: str) -> list[str]:
    """按 Base 图尺寸生成与训练集一致的 tile 类名。"""
    ys = build_axis_positions(img_h, TILE_SIZE, TILE_STRIDE)
    xs = build_axis_positions(img_w, TILE_SIZE, TILE_STRIDE)

    names = []
    for r, _y in enumerate(ys):
        for c, _x in enumerate(xs):
            names.append(f"{class_name}__r{r:02d}_c{c:02d}")
    return names


def create_error_dirs(source_dir: str, error_dir: str) -> None:
    """根据 source_dir 中的图片文件名，在 error_dir 下创建对应类别目录。"""
    src_path = Path(source_dir)
    err_path = Path(error_dir)

    if not src_path.exists():
        print(f"[Error] Source directory not found: '{source_dir}'")
        return

    err_path.mkdir(parents=True, exist_ok=True)

    files = list(src_path.glob("*.[pP][nN][gG]")) + list(src_path.glob("*.[jJ][pP][gG]"))
    if not files:
        print(f"[Warning] No source map images found in '{source_dir}'.")
        return

    created_count = 0
    exist_count = 0

    for file in files:
        class_name = file.stem

        if class_name in BASE_CLASS_NAMES:
            img = safe_imread(file)
            if img is None:
                print(f"[Warning] Failed to read {file}")
                continue
            h, w = img.shape[:2]
            class_names = enumerate_base_tile_names(h, w, class_name)
        else:
            class_names = [class_name]

        for name in class_names:
            target_dir = err_path / name
            if not target_dir.exists():
                target_dir.mkdir(parents=True)
                created_count += 1
            else:
                exist_count += 1

    for pending_base in BASE_CLASS_NAMES:
        pending_dir = err_path / "_pending" / pending_base
        pending_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 40)
    print(f"Source files detected   : {len(files)}")
    print(f"Directories created     : {created_count}")
    print(f"Directories skipped     : {exist_count}")
    print("=" * 40)
    print(f"Done. Base pending dirs created under '{error_dir}/_pending/'.")
    print("Base wrong images should go to error_images/_pending/Map01Base or Map02Base first.")


def parse_args() -> argparse.Namespace:
    """解析命令行覆盖项，并与 DEFAULT_OPTIONS 合并。"""
    parser = argparse.ArgumentParser(
        description="Initialize error sample directories from source image filenames"
    )
    parser.add_argument(
        "--source",
        default=argparse.SUPPRESS,
        help=(
            "Source directory containing original map images "
            f"(default: {DEFAULT_OPTIONS['source']})"
        ),
    )
    parser.add_argument(
        "--error",
        default=argparse.SUPPRESS,
        help=(
            "Root directory for hard negative sample subdirectories "
            f"(default: {DEFAULT_OPTIONS['error']})"
        ),
    )

    options = DEFAULT_OPTIONS.copy()
    options.update(vars(parser.parse_args()))
    return argparse.Namespace(**options)


if __name__ == "__main__":
    args = parse_args()
    create_error_dirs(args.source, args.error)
