"""从最新候选权重开始低学习率微调，并保留综合验证更优的权重。"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from ultralytics import YOLO
from ultralytics.nn.tasks import torch_safe_load

from train import DEFAULT_CONFIG, ValidationLossTrainer, find_latest_model

logger = logging.getLogger(__name__)

FINE_TUNE_CONFIG = {
    "epochs": 15,
    "patience": 5,
    "optimizer": "MuSGD",
    "lr0": 0.001,
    "lrf": 0.1,
    "warmup_epochs": 0.0,
}


def checkpoint_fitness(path: Path) -> float:
    """读取检查点保存的综合验证 fitness。"""
    checkpoint, _ = torch_safe_load(path)
    return float(checkpoint["train_metrics"]["fitness"])


def select_better_checkpoint(source: Path, candidate: Path) -> Path:
    """在粗训练与微调最优权重中选择综合验证更优者。"""
    source_fitness = checkpoint_fitness(source)
    candidate_fitness = checkpoint_fitness(candidate)
    winner = candidate if candidate_fitness > source_fitness else source
    selected = candidate.with_name("selected.pt")
    shutil.copyfile(winner, selected)
    logger.info(
        "[Fine-Tune] Selected %s (coarse=%.6f, fine=%.6f) -> %s",
        winner,
        source_fitness,
        candidate_fitness,
        selected,
    )
    return selected


def fine_tune(model_path: str | None = None) -> Path:
    """使用指定或自动发现的候选权重执行短程低学习率微调。"""
    resolved_path = model_path or find_latest_model(DEFAULT_CONFIG["project"])
    if not resolved_path:
        raise FileNotFoundError("No selected.pt or best.pt found for fine-tuning.")

    source = Path(resolved_path).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Fine-tune checkpoint does not exist: {source}")

    logger.info("[Fine-Tune] Starting from %s", source)
    model = YOLO(str(source))
    model.train(
        trainer=ValidationLossTrainer,
        data=DEFAULT_CONFIG["data"],
        imgsz=DEFAULT_CONFIG["imgsz"],
        batch=DEFAULT_CONFIG["batch"],
        workers=DEFAULT_CONFIG["workers"],
        device=DEFAULT_CONFIG["device"],
        erasing=DEFAULT_CONFIG["erasing"],
        auto_augment=DEFAULT_CONFIG["auto_augment"],
        save=True,
        project=DEFAULT_CONFIG["project"],
        name="finetune",
        **FINE_TUNE_CONFIG,
    )
    return select_better_checkpoint(source, Path(model.trainer.best))


if __name__ == "__main__":
    fine_tune()
