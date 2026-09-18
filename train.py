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
                    [--nbs <int>] [--workers <int>] [--patience <int>]
                    [--device <id>] [--name <str>] [--compile <mode|False>]
                    [--shutdown]

--shutdown：训练结束（早停、跑满轮数或异常退出）后执行系统关机，用于云 GPU 无人值守时省钱；
权重与 results.csv 在每轮结束时已落盘，关机前不再有待写数据。
"""

from __future__ import annotations

import argparse
import logging
import random
import subprocess
from copy import copy
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from ultralytics import YOLO
from ultralytics.data.augment import classify_augmentations
from ultralytics.models.yolo.classify import ClassificationTrainer, ClassificationValidator

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# 训练默认配置；命令行参数只作为覆盖项，不作为配置来源。
DEFAULT_CONFIG = {
    "data": "dataset",  # 数据集根目录
    "model": "auto",   # 权重路径，"auto" 表示自动发现最新历史权重
    "imgsz": 128,      # 训练输入图像尺寸（正方形边长）
    "batch": 128,      # 每步训练的样本数；保持八月稳定训练的更新密度
    "nbs": 64,         # 名义 batch；与稳定训练配置保持一致
    "workers": 24,     # DataLoader 并行工作线程数
    # 128px、batch 128 时每步瓶颈是 Python 逐个下发 GPU 算子：4090 实测 eager 53ms/步、GPU 占用 21%，
    # 数据加载 5850 张/秒并不缺。CUDA Graphs 把前向与反向降到约 31ms/步，batch 与迭代次数不变；
    # ultralytics 编译时训练集 drop_last，每轮少 1 步（375173 % 128 = 5 张随机样本）。
    "compile": "reduce-overhead",
    "patience": 20,    # 早停等待轮数（验证指标无提升时触发）
    "epochs": 200,     # 最大训练轮数
    "device": "0",     # CUDA 设备；可传 cpu
    "project": "runs/classify",
    "name": "train",
    "erasing": 0.0,
    "auto_augment": None,
}

FIXED_WORST_LOSS_WEIGHT = 0.25


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

        rows = []
        for probability, (_index, path, target) in zip(probabilities, fixed_samples):
            top_probabilities, top_indices = probability.topk(2)
            predicted, runner_up = map(int, top_indices)
            status = "OK" if predicted == target else "MISS"
            rows.append(
                (
                    status,
                    self.names[target],
                    Path(path).stem.removeprefix("fixed_"),
                    f"{probability[target]:.2%}",
                    self.names[predicted],
                    f"{top_probabilities[0]:.2%}",
                    self.names[runner_up],
                    f"{top_probabilities[1]:.2%}",
                    f"{(top_probabilities[0] - top_probabilities[1]) * 100:.2f}pp",
                )
            )

        headers = ("ST", "TARGET", "SAMPLE", "P(T)", "TOP1", "P1", "TOP2", "P2", "GAP")
        widths = [max(map(len, column)) for column in zip(headers, *rows)]
        numeric_columns = {3, 5, 7, 8}

        def format_row(row: tuple[str, ...]) -> str:
            return "  ".join(
                f"{value:>{width}}" if index in numeric_columns else f"{value:<{width}}"
                for index, (value, width) in enumerate(zip(row, widths))
            )

        correct_count = sum(row[0] == "OK" for row in rows)
        logger.info(
            f"[Fixed Val][{trainer.epoch + 1}/{trainer.epochs}] "
            f"{correct_count}/{len(rows)} OK"
        )
        logger.info(format_row(headers))
        logger.info("  ".join("-" * width for width in widths))
        for row in rows:
            logger.info(format_row(row))

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
            fixed_loss = (
                fixed_metrics["loss"]
                + FIXED_WORST_LOSS_WEIGHT * fixed_metrics["worst_loss"]
            )
            metrics.update(
                {
                    f"fixed_val/{name}": round(value, 5)
                    for name, value in fixed_metrics.items()
                }
            )

        metrics["fitness"] = 1.0 / (1.0 + metrics["val/loss"] + fixed_loss)
        return metrics


# 在线放大按 RandomResizedCrop 的面积取样：放大倍率 1/√a ∈ [1, 1.195]。
# 预处理已撤掉缩放扰动，这里是训练中唯一的放大来源。
ONLINE_ZOOM_AREA = (0.7, 1.0)
# 每轮随机平移的最大像素数。八月 RandomResizedCrop 的裁剪偏移最大约 19px，训出的模型最稳；
# 九月改成静态像素后，模型在实机渲染差异（滑索链、光晕、插值）上明显更敏感。
# 6px 小于均衡滑窗步长 8px 的一个 tile，Base 中心仍落在本 tile 内，Tier 中心几乎不会离开高亮区。
ONLINE_SHIFT_MAX = 6
# 不放大、整数像素平移的比例：这一支不经过双线性插值，像素原样进网络。
# 放大支路会把地图层 1–2px 的斑驳纹理抹平（高频 RMS 16.8→12.9，低于干净底图的 14.0），
# 而实机帧是未经插值的原始像素；保留一部分原样像素，模型才见得到实机强度的纹理。
ONLINE_EXACT_PROB = 0.4
# 镜像在实机不存在，但八月的翻转训练让模型学到与像素排布无关的结构特征：
# 翻转判别 Aug11 97%、Sep07 47%，稳的模型恰好是翻转不变的，所以按八月概率恢复。
ONLINE_HFLIP_PROB = 0.5
MINIMAP_CENTER = 64
MINIMAP_RADIUS = 53


class CenterZoom:
    """以玩家像素为基准的每轮随机放大、小幅平移与镜像，恢复八月加载裁剪提供的像素随机性。

    九月关闭 RandomResizedCrop 后数据集变成静态像素，模型只在模糊、缩放这类平滑扰动上崩溃。
    这里补回放大重采样、±ONLINE_SHIFT_MAX 像素平移和水平镜像；玩家指针随画面一起变换，
    与八月 RandomResizedCrop 的行为一致。变换后按小地图圆重新遮罩。
    """

    def __init__(
        self,
        area: tuple[float, float] = ONLINE_ZOOM_AREA,
        shift_max: int = ONLINE_SHIFT_MAX,
        hflip_prob: float = ONLINE_HFLIP_PROB,
        exact_prob: float = ONLINE_EXACT_PROB,
    ):
        self.area = area
        self.shift_max = shift_max
        self.hflip_prob = hflip_prob
        self.exact_prob = exact_prob
        # 与 preprocess.apply_minimap_mask 逐像素一致，变换后圆外溢出的地图内容重新清零。
        self.mask = np.zeros((MINIMAP_CENTER * 2, MINIMAP_CENTER * 2), dtype=np.uint8)
        cv2.circle(self.mask, (MINIMAP_CENTER, MINIMAP_CENTER), MINIMAP_RADIUS, 255, -1)

    def __call__(self, image: Image.Image) -> Image.Image:
        zoom = random.uniform(*self.area) ** -0.5
        dx = random.uniform(-self.shift_max, self.shift_max)
        dy = random.uniform(-self.shift_max, self.shift_max)
        flip = random.random() < self.hflip_prob
        if random.random() < self.exact_prob:
            zoom, dx, dy = 1.0, float(round(dx)), float(round(dy))
        return Image.fromarray(self.apply(np.asarray(image), zoom, dx, dy, flip))

    def apply(
        self,
        pixels: np.ndarray,
        zoom: float,
        dx: float = 0.0,
        dy: float = 0.0,
        flip: bool = False,
    ) -> np.ndarray:
        if flip:
            pixels = np.ascontiguousarray(pixels[:, ::-1])
        offset = MINIMAP_CENTER * (1.0 - zoom)
        matrix = np.float32([[zoom, 0.0, offset + dx], [0.0, zoom, offset + dy]])
        zoomed = cv2.warpAffine(
            pixels,
            matrix,
            (pixels.shape[1], pixels.shape[0]),
            flags=cv2.INTER_LINEAR,
        )
        return cv2.bitwise_and(zoomed, zoomed, mask=self.mask)


class ValidationLossTrainer(ClassificationTrainer):
    """让 best.pt 和早停由最低综合验证损失决定。"""

    def build_dataset(self, img_path, mode="train", batch=None):
        dataset = super().build_dataset(img_path, mode, batch)
        if mode == "train":
            # 放大、平移、镜像由 CenterZoom 统一处理，ultralytics 自带的裁剪与翻转关闭，长宽比拉伸不做。
            color_transforms = classify_augmentations(
                size=self.args.imgsz,
                scale=(1.0, 1.0),
                ratio=(1.0, 1.0),
                hflip=0.0,
                vflip=0.0,
                auto_augment=self.args.auto_augment,
                erasing=self.args.erasing,
                hsv_h=self.args.hsv_h,
                hsv_s=self.args.hsv_s,
                hsv_v=self.args.hsv_v,
            )
            dataset.torch_transforms = T.Compose([CenterZoom(), *color_transforms.transforms])
        return dataset

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

    try:
        model.train(
            trainer=ValidationLossTrainer,
            data=args.data,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            nbs=args.nbs,
            workers=args.workers,
            device=args.device,
            patience=args.patience,
            erasing=args.erasing,
            auto_augment=args.auto_augment,
            compile=args.compile,
            scale=0.0,
            fliplr=0.0,
            flipud=0.0,
            save=True,
            project=args.project,
            name=args.name,
        )
    finally:
        if getattr(args, "shutdown", False):
            shutdown_machine()


def shutdown_machine() -> None:
    """训练结束后关机。经 shell 调用：AutoDL 的 /usr/bin/shutdown 是没有 #! 行的脚本，直接 execve 会 ENOEXEC。"""
    logger.info("[Shutdown] Training finished, powering off the machine")
    for command in ("shutdown -h now", "shutdown"):
        try:
            if subprocess.run(command, shell=True, check=False).returncode == 0:
                return
        except OSError:
            continue
    logger.error("[Shutdown] shutdown command failed; machine is still running")


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
        "--nbs",
        type=int,
        default=argparse.SUPPRESS,
        help=f"Nominal batch size for optimizer scaling (default: {DEFAULT_CONFIG['nbs']})",
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

    parser.add_argument(
        "--compile",
        default=argparse.SUPPRESS,
        help=f"torch.compile mode passed to YOLO, or False (default: {DEFAULT_CONFIG['compile']})",
    )

    parser.add_argument(
        "--shutdown",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Power off the machine after training ends (early stop, max epochs or crash)",
    )

    config = DEFAULT_CONFIG.copy()
    config.update(vars(parser.parse_args()))
    if str(config["compile"]).lower() in {"false", "0", "none", "off"}:
        config["compile"] = False
    return argparse.Namespace(**config)


if __name__ == "__main__":
    train(parse_args())
