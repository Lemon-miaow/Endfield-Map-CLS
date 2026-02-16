import argparse
import logging
import math
import random
import shutil
from pathlib import Path

import cv2
import numpy as np

# Configuration
CONFIG = {
    "OUTPUT_SIZE": 128,  # Size of the output training images (128x128)
    "MASK_DIAMETER": 117,  # Diameter of the circular valid area in the minimap
    "TARGET_COUNT": 3000,  # Target number of samples to generate per class
    "VAL_RATIO": 0.2,  # Fraction of data to use for validation (20%)
    "STRIDE": 40,  # Step size for sliding window scanning over source images
    "STD_THRESHOLD": 5.0,  # Minimum standard deviation to consider a patch valid (filters empty areas)
    "OCCLUSION_COUNT": 2,  # Maximum number of random occlusion blocks to add
    "OCCLUSION_SIZE": 25,  # Maximum size of occlusion blocks
}

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    def __init__(self, input_dir: str, output_dir: str):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.train_dir = self.output_dir / "train"
        self.val_dir = self.output_dir / "val"
        self.safe_size = int(math.ceil(math.sqrt(2 * CONFIG["OUTPUT_SIZE"] ** 2)))

    def run(self):
        """Main execution pipeline."""
        if self.output_dir.exists():
            shutil.rmtree(self.output_dir)
        self.train_dir.mkdir(parents=True, exist_ok=True)
        self.val_dir.mkdir(parents=True, exist_ok=True)

        for file_path in self.input_dir.glob(
            "*.[pP][nN][gG]"
        ):  # Case insensitive globbing not native, simple workaround
            self._process_class(file_path)

        # Also check for jpg
        for file_path in self.input_dir.glob("*.[jJ][pP][gG]"):
            self._process_class(file_path)

        logger.info("Preprocessing completed successfully.")

    def _process_class(self, file_path: Path):
        class_name = file_path.stem
        logger.info(f"Processing class: {class_name}")

        # Create class directories
        (self.train_dir / class_name).mkdir(exist_ok=True)
        (self.val_dir / class_name).mkdir(exist_ok=True)

        # Load and normalize image
        img = self._load_image(file_path)
        if img is None:
            return

        # Generate samples
        samples = self._generate_samples(img)

        # Split and save
        self._save_dataset(samples, class_name)

    def _load_image(self, path: Path) -> np.ndarray:
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            logger.warning(f"Failed to load {path}")
            return None

        # Handle Alpha channel: transform transparent to black
        if img.ndim == 3 and img.shape[2] == 4:
            bgr = img[..., :3]
            alpha = img[..., 3]
            bgr[alpha < 10] = [0, 0, 0]
            img = bgr

        # Add padding
        pad = self.safe_size // 2
        return cv2.copyMakeBorder(
            img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=[0, 0, 0]
        )

    def _generate_samples(self, img: np.ndarray) -> list:
        h, w = img.shape[:2]
        pad = self.safe_size // 2
        orig_h, orig_w = h - 2 * pad, w - 2 * pad

        valid_centers = []
        samples = []

        # 1. Sliding window scanning
        for y in range(0, orig_h, CONFIG["STRIDE"]):
            for x in range(0, orig_w, CONFIG["STRIDE"]):
                cx, cy = x + pad, y + pad
                if self._is_valid(self._extract_roi(img, cx, cy, 0)):
                    valid_centers.append((cx, cy))

        if not valid_centers:
            logger.warning("No valid regions found.")
            return []

        # 2. Generate base samples
        for cx, cy in valid_centers:
            if len(samples) >= CONFIG["TARGET_COUNT"]:
                break
            samples.append(self._process_patch(img, cx, cy, 0))

        # 3. Augmentation to reach target
        while len(samples) < CONFIG["TARGET_COUNT"]:
            cx, cy = random.choice(valid_centers)
            # Random jitter and rotation
            nx = cx + random.randint(-5, 5)
            ny = cy + random.randint(-5, 5)
            angle = random.uniform(0, 360)

            patch = self._extract_roi(img, nx, ny, angle)
            if self._is_valid(patch):
                if random.random() > 0.5:
                    patch = cv2.flip(patch, random.choice([-1, 0, 1]))
                samples.append(self._augment_patch(patch))

        return samples

    def _extract_roi(self, img, cx, cy, angle):
        """Extracts rotated ROI from padded image."""
        half = self.safe_size // 2
        patch = img[cy - half : cy + half, cx - half : cx + half]

        if angle != 0:
            M = cv2.getRotationMatrix2D((half, half), angle, 1.0)
            patch = cv2.warpAffine(
                patch, M, (self.safe_size, self.safe_size), borderValue=(0, 0, 0)
            )

        start = (self.safe_size - CONFIG["OUTPUT_SIZE"]) // 2
        end = start + CONFIG["OUTPUT_SIZE"]
        return patch[start:end, start:end]

    def _is_valid(self, patch):
        """Checks if patch contains valid map data (variance check)."""
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        # Check center region (approx 30% of image)
        cy, cx = CONFIG["OUTPUT_SIZE"] // 2, CONFIG["OUTPUT_SIZE"] // 2
        r = 30
        center = gray[cy - r : cy + r, cx - r : cx + r]
        valid_pixels = center[center > 5]

        if len(valid_pixels) < (center.size * 0.15):
            return False
        return np.std(valid_pixels) > CONFIG["STD_THRESHOLD"]

    def _process_patch(self, img, cx, cy, angle):
        patch = self._extract_roi(img, cx, cy, angle)
        return self._augment_patch(patch)

    def _augment_patch(self, patch):
        """Applies mask and occlusion."""
        # Circular Mask
        mask = np.zeros((CONFIG["OUTPUT_SIZE"], CONFIG["OUTPUT_SIZE"]), dtype=np.uint8)
        cv2.circle(
            mask,
            (CONFIG["OUTPUT_SIZE"] // 2, CONFIG["OUTPUT_SIZE"] // 2),
            CONFIG["MASK_DIAMETER"] // 2,
            255,
            -1,
        )
        patch = cv2.bitwise_and(patch, patch, mask=mask)

        # Random Occlusion
        offset = (CONFIG["OUTPUT_SIZE"] - CONFIG["MASK_DIAMETER"]) // 2
        limit = CONFIG["OUTPUT_SIZE"] - offset

        for _ in range(random.randint(1, CONFIG["OCCLUSION_COUNT"])):
            w, h = (
                random.randint(10, CONFIG["OCCLUSION_SIZE"]),
                random.randint(10, CONFIG["OCCLUSION_SIZE"]),
            )
            x = random.randint(offset, limit - w)
            y = random.randint(offset, limit - h)

            if random.random() > 0.5:
                patch[y : y + h, x : x + w] = 0
            else:
                patch[y : y + h, x : x + w] = np.random.randint(
                    0, 256, (h, w, 3), dtype=np.uint8
                )

        return patch

    def _save_dataset(self, samples, class_name):
        random.shuffle(samples)
        split_idx = int(len(samples) * CONFIG["VAL_RATIO"])

        val_samples = samples[:split_idx]
        train_samples = samples[split_idx:]

        for i, img in enumerate(train_samples):
            cv2.imwrite(str(self.train_dir / class_name / f"{i:05d}.jpg"), img)

        for i, img in enumerate(val_samples):
            cv2.imwrite(str(self.val_dir / class_name / f"{i:05d}.jpg"), img)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dataset Preprocessing Pipeline")
    parser.add_argument("--input", default="source_images", help="Input directory")
    parser.add_argument("--output", default="dataset", help="Output directory")
    args = parser.parse_args()

    DataPreprocessor(args.input, args.output).run()
