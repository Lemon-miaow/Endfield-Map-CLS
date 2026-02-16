import argparse
import json
import logging
from pathlib import Path

from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def find_latest_model(base_dir="runs/classify") -> Path:
    """Automatically finds the latest 'best.pt' in the runs directory."""
    base_path = Path(base_dir)
    if not base_path.exists():
        raise FileNotFoundError(f"Directory not found: {base_dir}")

    candidates = list(base_path.rglob("weights/best.pt"))
    if not candidates:
        raise FileNotFoundError(f"No 'best.pt' found in {base_dir}")

    # Return the most recently modified file
    return max(candidates, key=lambda p: p.stat().st_mtime)


def export_model(model_path: str, imgsz: int):
    """Exports YOLO model to ONNX and saves class mapping to JSON."""

    # 1. Resolve Model Path
    if model_path:
        target_path = Path(model_path)
        if not target_path.exists():
            logger.error(f"Error: Model file does not exist at {target_path}")
            return
    else:
        logger.info("No model path provided. Searching for latest training run...")
        try:
            target_path = find_latest_model()
            logger.info(f"Found latest model: {target_path}")
        except FileNotFoundError as e:
            logger.error(f"Error: {e}")
            return

    # 2. Load and Export
    logger.info(f"Loading model...")
    model = YOLO(str(target_path))

    logger.info(f"Exporting to ONNX (Size: {imgsz}x{imgsz})...")
    export_filename = model.export(format="onnx", imgsz=imgsz)

    # 3. Save Class Mapping to JSON
    json_path = Path(export_filename).with_suffix(".json")
    names_list = [model.names[i] for i in range(len(model.names))]

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(names_list, f, indent=4, ensure_ascii=False)

    logger.info("=" * 50)
    logger.info(f"Export Success! File: {export_filename}")
    logger.info(f"Class mapping saved to: {json_path}")
    logger.info("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Export Tool")
    parser.add_argument(
        "--model",
        default=None,
        help="Path to .pt model (optional, defaults to latest run)",
    )
    parser.add_argument("--imgsz", type=int, default=128, help="Inference image size")

    args = parser.parse_args()
    export_model(args.model, args.imgsz)
