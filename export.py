"""
export.py — 模型导出脚本

将训练完成的 YOLO .pt 权重导出为 ONNX 格式，并生成配套的
部署元数据 JSON 文件，供 C++ 推理端（YoloPredictor）直接加载。

JSON 输出字段:
    input_name     ONNX 输入节点名称
    output_name    ONNX 输出节点名称
    classes        类别名称列表（索引与模型输出通道对应）
    region_mapping 区域映射表（由外部元数据注入，默认为空）

用法:
    python export.py [--model <path>] [--imgsz <int>] [--meta <path>]
"""

import argparse
import json
import logging
from pathlib import Path

from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def find_latest_model(base_dir: str = "runs/classify") -> Path:
    """在训练输出目录中查找修改时间最新的 best.pt 权重文件。

    Args:
        base_dir: 训练结果根目录，默认为 runs/classify。

    Returns:
        最新 best.pt 的 Path 对象。

    Raises:
        FileNotFoundError: 目录不存在或其中没有 best.pt 时抛出。
    """
    base_path = Path(base_dir)
    if not base_path.exists():
        raise FileNotFoundError(f"Directory not found: {base_dir}")

    candidates = list(base_path.rglob("weights/best.pt"))
    if not candidates:
        raise FileNotFoundError(f"No 'best.pt' found in {base_dir}")

    return max(candidates, key=lambda p: p.stat().st_mtime)


def export_model(model_path: str, imgsz: int, meta_config_path: str) -> None:
    """加载模型并导出为 ONNX，同时写出部署元数据 JSON。

    执行步骤：
        1. 加载外部元数据（region_mapping 等），缺失时使用默认值。
        2. 解析目标模型路径：显式指定优先，否则自动发现最新训练结果。
        3. 调用 YOLO.export() 导出 ONNX（opset 21）。
        4. 将类别列表与元数据合并写入同名 .json 文件。

    Args:
        model_path:      .pt 权重路径；传入空字符串或 None 时自动搜索。
        imgsz:           导出时指定的推理图像尺寸（正方形边长）。
        meta_config_path: 外部元数据 JSON 路径（包含 region_mapping 等字段）。
    """
    # 默认元数据，外部配置文件存在时增量覆盖
    meta_config = {
        "input_name": "images",
        "output_name": "output0",
        "region_mapping": {},
    }
    if meta_config_path:
        meta_path = Path(meta_config_path)
        if meta_path.exists():
            logger.info(f"Loading external metadata from: {meta_path}")
            with open(meta_path, "r", encoding="utf-8") as f:
                meta_config.update(json.load(f))
        else:
            logger.warning(
                f"Metadata config not found at {meta_path}, using default empty mapping."
            )

    # 解析目标模型路径
    if model_path:
        target_path = Path(model_path)
        if not target_path.exists():
            logger.error(f"Model file does not exist: {target_path}")
            return
    else:
        logger.info("No model path provided. Searching for latest training run...")
        try:
            target_path = find_latest_model()
            logger.info(f"Found latest model: {target_path}")
        except FileNotFoundError as e:
            logger.error(str(e))
            return

    logger.info("Loading model...")
    model = YOLO(str(target_path))

    logger.info(f"Exporting to ONNX (size: {imgsz}x{imgsz}, opset: 21)...")
    export_filename = model.export(format="onnx", imgsz=imgsz, opset=21)

    # 写出部署元数据 JSON（与 ONNX 文件同名同目录）
    json_path = Path(export_filename).with_suffix(".json")
    export_config = {
        "input_name": meta_config.get("input_name", "images"),
        "output_name": meta_config.get("output_name", "output0"),
        "classes": [model.names[i] for i in range(len(model.names))],
        "region_mapping": meta_config.get("region_mapping", {}),
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(export_config, f, indent=4, ensure_ascii=False)

    logger.info("=" * 50)
    logger.info(f"Export success. ONNX: {export_filename}")
    logger.info(f"Deploy config:  {json_path}")
    logger.info("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO Model Export Tool")
    parser.add_argument(
        "--model",
        default=None,
        help="Path to .pt weights file (optional, defaults to latest training run)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=128,
        help="Inference image size (default: 128)",
    )
    parser.add_argument(
        "--meta",
        default="deploy_meta.json",
        help="Path to external metadata config JSON (e.g., region_mapping)",
    )

    args = parser.parse_args()
    export_model(args.model, args.imgsz, args.meta)
