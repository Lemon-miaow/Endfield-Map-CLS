"""
init_error_dirs.py — 困难负样本目录初始化脚本

扫描 source_images 目录中的原始地图切片，在 error_images 目录下
为每个类别（文件名不含扩展名）创建同名空目录，用于后续手动投放
推理失败的截图，供 preprocess.py 的困难负样本过采样逻辑消费。

用法:
    python init_error_dirs.py [--source <dir>] [--error <dir>]
"""

import argparse
from pathlib import Path


def create_error_dirs(source_dir: str, error_dir: str) -> None:
    """根据 source_dir 中的图片文件名，在 error_dir 下创建对应的类别子目录。

    若子目录已存在则跳过（exist_ok 语义），仅创建缺失的目录。
    执行完毕后打印汇总报告。

    Args:
        source_dir: 原始地图切片根目录，图片文件名即为类别名。
        error_dir:  困难负样本根目录，将在此处创建类别子目录。
    """
    src_path = Path(source_dir)
    err_path = Path(error_dir)

    if not src_path.exists():
        print(f"[Error] Source directory not found: '{source_dir}'")
        return

    err_path.mkdir(parents=True, exist_ok=True)

    # 收集所有合法图片文件（兼容大小写后缀）
    files = list(src_path.glob("*.[pP][nN][gG]")) + list(src_path.glob("*.[jJ][pP][gG]"))

    if not files:
        print(f"[Warning] No source map images found in '{source_dir}'.")
        return

    created_count = 0
    exist_count = 0

    for file in files:
        class_name = file.stem  # 文件名（不含扩展名）对应类别名
        target_dir = err_path / class_name

        if not target_dir.exists():
            target_dir.mkdir(parents=True)
            created_count += 1
        else:
            exist_count += 1

    print("=" * 40)
    print(f"Total classes detected : {len(files)}")
    print(f"Directories created    : {created_count}")
    print(f"Directories skipped    : {exist_count}")
    print("=" * 40)
    print(f"Done. Place inference failure screenshots under '{error_dir}/<class_name>/'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Initialize error sample directories from source image filenames"
    )
    parser.add_argument(
        "--source",
        default="source_images",
        help="Source directory containing original map images (default: source_images)",
    )
    parser.add_argument(
        "--error",
        default="error_images",
        help="Root directory for hard negative sample subdirectories (default: error_images)",
    )

    args = parser.parse_args()
    create_error_dirs(args.source, args.error)
