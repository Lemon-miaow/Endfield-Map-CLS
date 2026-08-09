"""
train.py — YOLO 分类器训练脚本

支持两种启动模式：
    auto（默认）  自动查找 runs/classify 下修改时间最新的 selected.pt / best.pt
                  作为增量微调起点；若不存在历史权重则从 yolo26s-cls.pt 底模开始训练。
    显式指定      通过 --model 传入具体 .pt 路径，强制使用该权重初始化。

训练期间按生成验证集与固定真实验证集的综合损失保存 best.pt 和执行早停，
使真实场景中的正确类别置信度退化能够参与选优。

用法:
    python train.py [--data <dir>] [--model <path|auto>]
                    [--epochs <int>] [--imgsz <int>] [--batch <int>]
                    [--workers <int>] [--patience <int>]
                    [--device <id>] [--name <str>]
"""

from __future__ import annotations

import argparse
import logging
from copy import copy
from pathlib import Path

import torch
import torch.nn.functional as F
from ultralytics import YOLO
from ultralytics.models.yolo.classify import ClassificationTrainer, ClassificationValidator

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# 训练默认配置；命令行参数只作为覆盖项，不作为配置来源。
DEFAULT_CONFIG = {
    "data": "dataset",  # 数据集根目录
    "model": "auto",   # 权重路径，"auto" 表示自动发现最新历史权重
    "imgsz": 128,      # 训练输入图像尺寸（正方形边长）
    "batch": 128,      # 每步训练的样本数
    "workers": 24,     # DataLoader 并行工作线程数
    "patience": 20,    # 早停等待轮数（验证指标无提升时触发）
    "epochs": 200,     # 最大训练轮数
    "device": "0",     # CUDA 设备；可传 cpu
    "project": "runs/classify",
    "name": "train",
    "erasing": 0.0,
    "auto_augment": None,
}


class ValidationLossValidator(ClassificationValidator):
    """使用生成验证集与固定真实验证集的综合损失选择检查点。"""

    def _evaluate_fixed_predictions(self, trainer) -> dict[str, float] | None:
        """独立计算固定验证图损失，并输出每张图的目标类别置信度。"""
        if getattr(trainer, "rank", -1) not in {-1, 0}:
            return None

        dataset = self.dataloader.dataset
        fixed_samples = [
            (index, path, target)
            for index, (path, target, *_rest) in enumerate(dataset.samples)
            if Path(path).name.startswith("fixed_")
        ]
        if not fixed_samples:
            return None

        images = torch.stack([dataset[index]["img"] for index, *_ in fixed_samples])
        model = trainer.ema.ema or trainer.model
        if trainer.args.compile and hasattr(model, "_orig_mod"):
            model = model._orig_mod

        with torch.inference_mode():
            output = model(images.to(self.device).float())
            if isinstance(output, (tuple, list)):
                probabilities, logits = output
            else:
                logits = output
                probabilities = logits.softmax(1)
            targets = torch.tensor(
                [target for _index, _path, target in fixed_samples],
                device=logits.device,
            )
            losses = F.cross_entropy(logits, targets, reduction="none")
            probabilities = probabilities.cpu()

        for probability, (_index, path, target) in zip(probabilities, fixed_samples):
            predicted = int(probability.argmax())
            status = "OK" if predicted == target else "MISS"
            logger.info(
                f"[Fixed Val][{trainer.epoch + 1}/{trainer.epochs}] "
                f"{self.names[target]}/{Path(path).name} {status}: "
                f"target={probability[target]:.2%}, "
                f"top1={self.names[predicted]} {probability[predicted]:.2%}"
            )

        top1_accuracy = float(
            (probabilities.argmax(1) == targets.cpu()).float().mean()
        )
        return {
            "loss": float(losses.mean()),
            "worst_loss": float(losses.max()),
            "top1_acc": top1_accuracy,
        }

    def __call__(self, trainer=None, model=None):
        metrics = super().__call__(trainer, model)
        if not isinstance(metrics, dict) or "val/loss" not in metrics:
            return metrics

        fixed_metrics = (
            self._evaluate_fixed_predictions(trainer)
            if trainer is not None
            else None
        )
        fixed_loss = 0.0
        if fixed_metrics is not None:
            fixed_loss = fixed_metrics["loss"]
            metrics.update(
                {
                    f"fixed_val/{name}": round(value, 5)
                    for name, value in fixed_metrics.items()
                }
            )

        metrics["fitness"] = 1.0 / (1.0 + metrics["val/loss"] + fixed_loss)
        return metrics


class ValidationLossTrainer(ClassificationTrainer):
    """让 best.pt 和早停由最低综合验证损失决定。"""

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        loader = super().get_dataloader(
            dataset_path,
            batch_size=batch_size,
            rank=rank,
            mode=mode,
        )
        if mode != "train" and self.args.compile:
            loader.batch_sampler.sampler.drop_last = False
        return loader

    def get_validator(self):
        self.loss_names = ["loss"]
        return ValidationLossValidator(
            self.test_loader,
            self.save_dir,
            args=copy(self.args),
            _callbacks=self.callbacks,
        )


def find_latest_model(base_dir: str = "runs/classify") -> str | None:
    """在训练输出目录中查找修改时间最新的候选权重。

    Args:
        base_dir: 训练结果根目录，默认为 runs/classify。

    Returns:
        最新 selected.pt 或 best.pt 的字符串路径；无候选文件时返回 None。
    """
    base_path = Path(base_dir)
    if not base_path.exists():
        return None

    candidates = [
        *base_path.rglob("weights/best.pt"),
        *base_path.rglob("weights/selected.pt"),
    ]
    if not candidates:
        return None

    latest = max(
        candidates,
        key=lambda path: (path.stat().st_mtime_ns, path.name == "selected.pt"),
    )
    return str(latest)


def train(args: argparse.Namespace) -> None:
    """执行 YOLO 分类器训练流程。

    根据 args.model 的值决定权重初始化方式（auto / 显式路径），
    然后调用 YOLO.train() 启动训练，结果保存至 runs/classify/<name>。

    Args:
        args: 由 argparse 解析的命令行参数对象。
    """
    model_path = args.model

    if model_path == "auto":
        latest_pt = find_latest_model(args.project)
        if latest_pt:
            logger.info(
                "[Auto-Detect] Found latest weights, "
                f"resuming incremental fine-tuning: {latest_pt}"
            )
            model_path = latest_pt
        else:
            logger.info(
                "[Auto-Detect] No previous weights found. "
                "Starting from base model: yolo26s-cls.pt"
            )
            model_path = "yolo26s-cls.pt"
    else:
        logger.info(f"Using specified weights: {model_path}")

    model = YOLO(model_path)

    model.train(
        trainer=ValidationLossTrainer,
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        patience=args.patience,
        erasing=args.erasing,
        auto_augment=args.auto_augment,
        save=True,
        project=args.project,
        name=args.name,
    )


def parse_args() -> argparse.Namespace:
    """解析命令行覆盖项，并与 DEFAULT_CONFIG 合并。"""
    parser = argparse.ArgumentParser(description="YOLO Classification Training Script")
    parser.add_argument(
        "--data",
        default=argparse.SUPPRESS,
        help=f"Path to dataset root directory (default: {DEFAULT_CONFIG['data']})",
    )
    parser.add_argument(
        "--model",
        default=argparse.SUPPRESS,
        help=f"Model weights path, or 'auto' (default: {DEFAULT_CONFIG['model']})",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=argparse.SUPPRESS,
        help=f"Number of training epochs (default: {DEFAULT_CONFIG['epochs']})",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=argparse.SUPPRESS,
        help=f"Input image size (default: {DEFAULT_CONFIG['imgsz']})",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=argparse.SUPPRESS,
        help=f"Batch size (default: {DEFAULT_CONFIG['batch']})",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=argparse.SUPPRESS,
        help=f"Number of DataLoader workers (default: {DEFAULT_CONFIG['workers']})",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=argparse.SUPPRESS,
        help=f"Early stopping patience in epochs (default: {DEFAULT_CONFIG['patience']})",
    )
    parser.add_argument(
        "--device",
        default=argparse.SUPPRESS,
        help=f"CUDA device index or ID (default: {DEFAULT_CONFIG['device']})",
    )
    parser.add_argument(
        "--name",
        default=argparse.SUPPRESS,
        help=f"Experiment name for output directory (default: {DEFAULT_CONFIG['name']})",
    )
    parser.add_argument(
        "--project",
        default=argparse.SUPPRESS,
        help=f"Training output root directory (default: {DEFAULT_CONFIG['project']})",
    )
    parser.add_argument(
        "--erasing",
        type=float,
        default=argparse.SUPPRESS,
        help=f"Random erasing strength passed to YOLO (default: {DEFAULT_CONFIG['erasing']})",
    )
    parser.add_argument(
        "--auto-augment",
        dest="auto_augment",
        default=argparse.SUPPRESS,
        help="YOLO auto augment policy override (default: disabled)",
    )

    config = DEFAULT_CONFIG.copy()
    config.update(vars(parser.parse_args()))
    return argparse.Namespace(**config)


if __name__ == "__main__":
    train(parse_args())
