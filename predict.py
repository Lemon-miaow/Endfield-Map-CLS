import argparse
import logging
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# Configuration
CONFIG = {
    "OUTPUT_SIZE": 128,  # The dimensions (width/height) of the output image after preprocessing
    "MASK_DIAMETER": 106,  # Diameter of the circular mask applied to the center of the image
    "GAME_RES_H": 720,  # Height of the game resolution the input screenshot is taken from
    "TARGET_RES_H": 720,  # Target height resolution to scale the input image to before processing
}

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


class Predictor:
    def __init__(self, model_path: str = None):
        self.model_path = self._resolve_model_path(model_path)
        logger.info(f"Loading model: {self.model_path}")
        self.model = YOLO(str(self.model_path))
        self.scale_ratio = CONFIG["TARGET_RES_H"] / CONFIG["GAME_RES_H"]

    def _resolve_model_path(self, model_path: str) -> Path:
        """Determines the model path, auto-discovering the latest run if necessary."""
        if model_path:
            path = Path(model_path)
            if not path.exists():
                raise FileNotFoundError(f"Specified model not found: {path}")
            return path

        # Auto-discovery logic
        runs_dir = Path("runs/classify")
        if not runs_dir.exists():
            raise FileNotFoundError(
                "No training runs found in 'runs/classify'. Please train a model first or specify a path."
            )

        # Find all 'best.pt' files and sort by modification time (newest first)
        candidates = list(runs_dir.rglob("weights/best.pt"))
        if not candidates:
            raise FileNotFoundError("No 'best.pt' found in training runs.")

        latest_model = max(candidates, key=lambda p: p.stat().st_mtime)
        logger.info(f"Auto-detected latest model: {latest_model}")
        return latest_model

    def preprocess(self, img: np.ndarray) -> np.ndarray:
        """Resizes, crops, and masks the input image."""
        # 1. Resize
        if self.scale_ratio != 1.0:
            h, w = img.shape[:2]
            img = cv2.resize(
                img,
                (int(w * self.scale_ratio), int(h * self.scale_ratio)),
                interpolation=cv2.INTER_AREA,
            )

        # 2. Center Crop to Canvas
        h, w = img.shape[:2]
        size = CONFIG["OUTPUT_SIZE"]
        canvas = np.zeros((size, size, 3), dtype=np.uint8)

        dst_x = max(0, (size - w) // 2)
        dst_y = max(0, (size - h) // 2)
        src_x = max(0, (w - size) // 2)
        src_y = max(0, (h - size) // 2)

        copy_w = min(w, size, size - dst_x, w - src_x)
        copy_h = min(h, size, size - dst_y, h - src_y)

        canvas[dst_y : dst_y + copy_h, dst_x : dst_x + copy_w] = img[
            src_y : src_y + copy_h, src_x : src_x + copy_w
        ]

        # 3. Apply Circular Mask
        mask = np.zeros((size, size), dtype=np.uint8)
        cv2.circle(mask, (size // 2, size // 2), CONFIG["MASK_DIAMETER"] // 2, 255, -1)

        return cv2.bitwise_and(canvas, canvas, mask=mask)

    def predict(self, image_path: str, save_debug: bool = False):
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")

        img = cv2.imread(str(path))
        if img is None:
            raise ValueError(f"Failed to read image: {path}")

        processed = self.preprocess(img)

        if save_debug:
            debug_path = "debug_inference.jpg"
            cv2.imwrite(debug_path, processed)
            logger.info(f"Debug image saved to: {debug_path}")

        results = self.model(processed, verbose=False)

        # Return top result
        top1 = results[0].probs.top1
        return results[0].names[top1], results[0].probs.top1conf.item()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference Script")
    parser.add_argument("image", help="Path to test image")
    parser.add_argument(
        "--model", default=None, help="Model path (optional, defaults to latest run)"
    )
    parser.add_argument(
        "--debug", action="store_true", help="Save preprocessed debug image"
    )

    args = parser.parse_args()

    try:
        engine = Predictor(args.model)
        name, conf = engine.predict(args.image, args.debug)
        print(f"\n>>> Prediction: {name} ({conf:.2%})\n")
    except Exception as e:
        logger.error(f"Error: {e}")
