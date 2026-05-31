"""
preprocess_roi_auto_res.py — 截图小地图 ROI 批处理工具

将全屏截图按 720p 基准缩放，裁出小地图 ROI，居中放入 128x128 画布并应用圆形
mask。默认处理 raw_failures/ 到 processed_errors/，命令行参数只作为覆盖项。

用法:
    python preprocess_roi_auto_res.py [--input <path>] [--output <dir>]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


CONFIG = {
    "ROI_X": 49,
    "ROI_Y": 51,
    "ROI_W": 118,
    "ROI_H": 120,
    "OUTPUT_SIZE": 128,
    "MASK_DIAMETER": 106,
    "TARGET_RES_H": 720,
}
DEFAULT_OPTIONS = {
    "input": "raw_failures",
    "output": "processed_errors",
}
IMAGE_PATTERNS = (
    "*.[pP][nN][gG]",
    "*.[jJ][pP][gG]",
    "*.[jJ][pP][eE][gG]",
    "*.[bB][mM][pP]",
)


def safe_imread(path: Path):
    """读取图片，兼容 Path 对象和非 ASCII 文件名。"""
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_UNCHANGED)


def safe_imwrite(path: Path, img):
    """写入图片，自动创建父目录。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(path.suffix or ".png", img)
    if ok:
        buf.tofile(str(path))
    return ok


def process_image(img: np.ndarray) -> np.ndarray:
    """将一张全屏截图转换为 128x128 小地图 ROI。"""
    h, w = img.shape[:2]
    scale = CONFIG["TARGET_RES_H"] / float(h)
    if abs(scale - 1.0) > 1e-6:
        img = cv2.resize(
            img,
            (int(round(w * scale)), CONFIG["TARGET_RES_H"]),
            interpolation=cv2.INTER_AREA,
        )

    img_h, img_w = img.shape[:2]
    x1 = max(0, min(CONFIG["ROI_X"], img_w))
    y1 = max(0, min(CONFIG["ROI_Y"], img_h))
    x2 = max(0, min(CONFIG["ROI_X"] + CONFIG["ROI_W"], img_w))
    y2 = max(0, min(CONFIG["ROI_Y"] + CONFIG["ROI_H"], img_h))
    minimap = img[y1:y2, x1:x2]

    if minimap.ndim == 2:
        minimap = cv2.cvtColor(minimap, cv2.COLOR_GRAY2BGR)
    elif minimap.ndim == 3 and minimap.shape[2] == 4:
        minimap = cv2.cvtColor(minimap, cv2.COLOR_BGRA2BGR)

    size = CONFIG["OUTPUT_SIZE"]
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    mh, mw = minimap.shape[:2]
    if mh == 0 or mw == 0:
        return canvas

    dx = max(0, (size - mw) // 2)
    dy = max(0, (size - mh) // 2)
    sx = max(0, (mw - size) // 2)
    sy = max(0, (mh - size) // 2)
    cw = min(mw - sx, size - dx)
    ch = min(mh - sy, size - dy)
    canvas[dy : dy + ch, dx : dx + cw] = minimap[sy : sy + ch, sx : sx + cw]

    mask = np.zeros((size, size), dtype=np.uint8)
    cv2.circle(mask, (size // 2, size // 2), CONFIG["MASK_DIAMETER"] // 2, 255, -1)
    return cv2.bitwise_and(canvas, canvas, mask=mask)


def collect_input_files(input_path: Path) -> list[Path]:
    """收集输入文件，支持单文件或目录递归。"""
    if input_path.is_file():
        return [input_path]
    if not input_path.exists():
        return []

    files: list[Path] = []
    for pattern in IMAGE_PATTERNS:
        files.extend(input_path.rglob(pattern))
    return sorted(files)


def batch_process(input_path: str, output_dir: str) -> None:
    """批量处理截图并保持相对目录结构输出。"""
    in_path = Path(input_path)
    out_dir = Path(output_dir)
    files = collect_input_files(in_path)
    if not files:
        print(f"[WARN] no input images found: {in_path}")
        return

    for file_path in files:
        img = safe_imread(file_path)
        if img is None:
            print("[WARN] unreadable", file_path)
            continue

        out = process_image(img)
        rel_path = Path(file_path.name) if in_path.is_file() else file_path.relative_to(in_path)
        output_path = out_dir / rel_path
        safe_imwrite(output_path, out)
        print("[OK]", output_path)


def parse_args() -> argparse.Namespace:
    """解析命令行覆盖项，并与 DEFAULT_OPTIONS 合并。"""
    parser = argparse.ArgumentParser(description="Auto-adaptive screenshot to 128 minimap ROI")
    parser.add_argument(
        "-i",
        "--input",
        default=argparse.SUPPRESS,
        help=f"input image or directory (default: {DEFAULT_OPTIONS['input']})",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=argparse.SUPPRESS,
        help=f"output directory (default: {DEFAULT_OPTIONS['output']})",
    )

    options = DEFAULT_OPTIONS.copy()
    options.update(vars(parser.parse_args()))
    return argparse.Namespace(**options)


if __name__ == "__main__":
    args = parse_args()
    batch_process(args.input, args.output)
