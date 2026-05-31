"""
assign_error_tiles.py — Base 错图自动分配工具

将 error_images/_pending/Map01Base 和 Map02Base 中的错图交给当前分类模型推理，
按置信度分配到对应 Base tile 类目录；低置信度或异常样本进入 _review。

用法:
    python assign_error_tiles.py [--error <dir>] [--model <path>] [--threshold <float>]
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from predict import Predictor


BASE_CLASS_NAMES = {"Map01Base", "Map02Base"}
IMAGE_PATTERNS = ("*.[pP][nN][gG]", "*.[jJ][pP][gG]")
DEFAULT_OPTIONS = {
    "error": "error_images",
    "model": None,
    "threshold": 0.90,
}


def is_base_tile_class(name: str, base_name: str) -> bool:
    """判断分类名是否属于指定 Base 大图的 tile 类。"""
    return name.startswith(base_name + "__r")


def iter_images(directory: Path) -> list[Path]:
    """返回目录下可处理图片，保持稳定顺序。"""
    files: list[Path] = []
    for pattern in IMAGE_PATTERNS:
        files.extend(directory.glob(pattern))
    return sorted(files)


def select_target_class(
    predictions: list[tuple[str, float]],
    base_name: str,
) -> tuple[str | None, float]:
    """从预测结果中取第一个属于该 Base 的 tile 类。"""
    for name, confidence in predictions:
        if is_base_tile_class(name, base_name):
            return name, confidence
    return None, 0.0


def assign_pending_errors(error_dir: str, model_path: str | None, threshold: float) -> None:
    """处理 pending 目录，将 Base 错图复制到 tile 类目录或 review 目录。"""
    root = Path(error_dir)
    pending_root = root / "_pending"

    if not pending_root.exists():
        print(f"[Info] No pending dir found: {pending_root}")
        return

    predictor = Predictor(model_path)

    for base_name in BASE_CLASS_NAMES:
        src_dir = pending_root / base_name
        if not src_dir.exists():
            continue

        files = iter_images(src_dir)
        if not files:
            continue

        review_dir = root / "_review" / base_name
        review_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nProcessing pending errors for {base_name}: {len(files)} files")

        for file_path in files:
            try:
                preds = predictor.predict(str(file_path), save_debug=False)
                target_class, target_conf = select_target_class(preds, base_name)

                if target_class is not None and target_conf >= threshold:
                    dst_dir = root / target_class
                    dst_dir.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(file_path, dst_dir / file_path.name)
                    print(f"[OK] {file_path.name} -> {target_class} ({target_conf:.2%})")
                else:
                    shutil.copy2(file_path, review_dir / file_path.name)
                    print(f"[REVIEW] {file_path.name}")

            except Exception as e:
                review_dir = root / "_review" / base_name
                review_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(file_path, review_dir / file_path.name)
                print(f"[ERR] {file_path.name}: {e}")

        for file_path in files:
            try:
                file_path.unlink()
            except OSError:
                pass


def parse_args() -> argparse.Namespace:
    """解析命令行覆盖项，并与 DEFAULT_OPTIONS 合并。"""
    parser = argparse.ArgumentParser(description="Assign pending base error images into tile classes")
    parser.add_argument(
        "--error",
        default=argparse.SUPPRESS,
        help=f"error_images root (default: {DEFAULT_OPTIONS['error']})",
    )
    parser.add_argument(
        "--model",
        default=argparse.SUPPRESS,
        help="model path override (default: latest training run)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=argparse.SUPPRESS,
        help=f"confidence threshold (default: {DEFAULT_OPTIONS['threshold']})",
    )

    options = DEFAULT_OPTIONS.copy()
    options.update(vars(parser.parse_args()))
    return argparse.Namespace(**options)


if __name__ == "__main__":
    args = parse_args()
    assign_pending_errors(args.error, args.model, args.threshold)
