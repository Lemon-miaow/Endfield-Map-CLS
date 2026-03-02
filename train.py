"""
train.py — YOLO 分类器训练脚本

支持两种启动模式：
    auto（默认）  自动查找 runs/classify 下修改时间最新的 best.pt
                  作为增量微调起点；若不存在历史权重则从 yolo26s-cls.pt 底模开始训练。
    显式指定      通过 --model 传入具体 .pt 路径，强制使用该权重初始化。

用法:
    python train.py [--data <dir>] [--model <path|auto>]
                    [--epochs <int>] [--imgsz <int>] [--batch <int>]
                    [--workers <int>] [--patience <int>]
                    [--device <id>] [--name <str>]
"""

import argparse
import logging
from pathlib import Path

from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# 训练默认超参数（可通过 CLI 参数覆盖）
DEFAULT_CONFIG = {
    "model": "auto",   # 权重路径，"auto" 表示自动发现最新历史权重
    "imgsz": 128,      # 训练输入图像尺寸（正方形边长）
    "batch": 128,      # 每步训练的样本数
    "workers": 6,      # DataLoader 并行工作线程数
    "patience": 20,    # 早停等待轮数（验证指标无提升时触发）
    "epochs": 200,     # 最大训练轮数
}


def find_latest_model(base_dir: str = "runs/classify") -> str | None:
    """在训练输出目录中查找修改时间最新的 best.pt 权重文件。

    Args:
        base_dir: 训练结果根目录，默认为 runs/classify。

    Returns:
        最新 best.pt 的字符串路径；目录不存在或无候选文件时返回 None。
    """
    base_path = Path(base_dir)
    if not base_path.exists():
        return None

    candidates = list(base_path.rglob("weights/best.pt"))
    if not candidates:
        return None

    return str(max(candidates, key=lambda p: p.stat().st_mtime))


def train(args: argparse.Namespace) -> None:
    """执行 YOLO 分类器训练流程。

    根据 args.model 的值决定权重初始化方式（auto / 显式路径），
    然后调用 YOLO.train() 启动训练，结果保存至 runs/classify/<name>。

    Args:
        args: 由 argparse 解析的命令行参数对象。
    """
    model_path = args.model

    if model_path == "auto":
        latest_pt = find_latest_model()
        if latest_pt:
            logger.info(f"[Auto-Detect] Found latest weights, resuming incremental fine-tuning: {latest_pt}")
            model_path = latest_pt
        else:
            logger.info("[Auto-Detect] No previous weights found. Starting from base model: yolo26s-cls.pt")
            model_path = "yolo26s-cls.pt"
    else:
        logger.info(f"Using specified weights: {model_path}")

    model = YOLO(model_path)

    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        patience=args.patience,
        erasing=0.6,
        auto_augment="randaugment",
        save=True,
        project="runs/classify",
        name=args.name or "train",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO Classification Training Script")
    parser.add_argument(
        "--data",
        default="dataset",
        help="Path to dataset root directory (default: dataset)",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_CONFIG["model"],
        help="Model weights path, or 'auto' to resume from latest run (default: auto)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_CONFIG["epochs"],
        help=f"Number of training epochs (default: {DEFAULT_CONFIG['epochs']})",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=DEFAULT_CONFIG["imgsz"],
        help=f"Input image size (default: {DEFAULT_CONFIG['imgsz']})",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=DEFAULT_CONFIG["batch"],
        help=f"Batch size (default: {DEFAULT_CONFIG['batch']})",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_CONFIG["workers"],
        help=f"Number of DataLoader workers (default: {DEFAULT_CONFIG['workers']})",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=DEFAULT_CONFIG["patience"],
        help=f"Early stopping patience in epochs (default: {DEFAULT_CONFIG['patience']})",
    )
    parser.add_argument(
        "--device",
        default="0",
        help="CUDA device index or ID (e.g. 0, 0,1, cpu)",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Experiment name for the training run output directory",
    )

    args = parser.parse_args()
    train(args)
