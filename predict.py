"""
predict.py — 单图推理脚本

对输入截图执行与 C++ 推理端等价的预处理流程，调用 YOLO 分类模型
输出各类别置信度，并按置信度降序打印结果。

预处理流程（与 C++ 两阶段等价）:
    1. 按分辨率缩放比例将截图缩放至 720p 基准。
    2. 按固定坐标裁取小地图 ROI（与调用层 MapLocator 逻辑一致）。
    3. 将 ROI 居中放置于 OUTPUT_SIZE×OUTPUT_SIZE 黑色画布。
    4. 应用半径为 MASK_DIAMETER/2 的圆形 Mask，消除外框噪声。

用法:
    python predict.py <image_path> [--model <path>] [--debug]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# 推理预处理规格（与训练集 / C++ 推理端保持一致）
CONFIG = {
    "OUTPUT_SIZE": 128,    # 预处理输出图像的边长（像素）
    "MASK_DIAMETER": 106,  # 圆形 Mask 的直径
    "GAME_RES_H": 720,     # 采集截图时的游戏分辨率高度
    "TARGET_RES_H": 720,   # 推理前期望缩放到的目标分辨率高度
    # 小地图在 720p 截图中的 ROI 坐标（与 C++ MapLocator 保持一致）
    "ROI_X": 49,
    "ROI_Y": 51,
    "ROI_W": 118,
    "ROI_H": 120,
}

DEFAULT_OPTIONS = {
    "model": None,
    "debug": False,
    "debug_path": "debug_inference.jpg",
}

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def safe_imread(path, flags=cv2.IMREAD_COLOR) -> np.ndarray:
    """读取图片，兼容路径中包含非 ASCII 字符（如中文）的情况。"""
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), flags)


class Predictor:
    """YOLO 分类推理器。

    封装模型加载、预处理和推理全流程，提供与 C++ 两阶段推理等价的
    完整预处理逻辑（缩放 → 裁取 ROI → 居中 → 圆形 Mask）。

    Args:
        model_path: .pt 或 .onnx 权重路径；为 None 时自动发现最新训练结果。
    """

    def __init__(self, model_path: str = None):
        self.model_path = self._resolve_model_path(model_path)
        logger.info(f"Loading model: {self.model_path}")
        self.model = YOLO(str(self.model_path))

    def _resolve_model_path(self, model_path: str) -> Path:
        """解析模型路径，未指定时自动发现最新的训练结果。

        Args:
            model_path: 用户显式指定的路径，或 None。

        Returns:
            有效的模型文件 Path 对象。

        Raises:
            FileNotFoundError: 指定路径不存在，或自动搜索无结果时抛出。
        """
        if model_path:
            path = Path(model_path)
            if not path.exists():
                raise FileNotFoundError(f"Specified model not found: {path}")
            return path

        runs_dir = Path("runs/classify")
        if not runs_dir.exists():
            raise FileNotFoundError(
                "No training runs found in 'runs/classify'. "
                "Please train a model first or specify a path with --model."
            )

        candidates = list(runs_dir.rglob("weights/best.pt"))
        if not candidates:
            raise FileNotFoundError("No 'best.pt' found in training runs.")

        latest_model = max(candidates, key=lambda p: p.stat().st_mtime)
        logger.info(f"Auto-detected latest model: {latest_model}")
        return latest_model

    def preprocess(self, img: np.ndarray) -> np.ndarray:
        """对原始全屏截图执行完整预处理，与 C++ 两阶段流程等价。

        处理顺序：
            1. 若游戏分辨率非 720p，等比缩放至 720p 基准。
            2. 按固定坐标裁取小地图 ROI（与 C++ MapLocator 逻辑一致）。
            3. 将 ROI 居中放置于 128×128 黑色画布。
            4. 应用圆形 Mask，消除小地图外框噪声。

        Args:
            img: 原始 BGR 全屏截图。

        Returns:
            经预处理的 OUTPUT_SIZE×OUTPUT_SIZE BGR 图像。
        """
        # 阶段一：按输入截图的实际高度自动缩放到 720p 基准
        h, w = img.shape[:2]
        scale_ratio = CONFIG["TARGET_RES_H"] / float(h)
        if abs(scale_ratio - 1.0) > 1e-6:
            img = cv2.resize(
                img,
                (int(round(w * scale_ratio)), CONFIG["TARGET_RES_H"]),
                interpolation=cv2.INTER_AREA,
            )

        # 阶段二：裁取小地图 ROI，并做边界钳位防止越界
        img_h, img_w = img.shape[:2]
        x, y = CONFIG["ROI_X"], CONFIG["ROI_Y"]
        rw, rh = CONFIG["ROI_W"], CONFIG["ROI_H"]
        roi_x1 = max(0, min(x, img_w))
        roi_y1 = max(0, min(y, img_h))
        roi_x2 = max(0, min(x + rw, img_w))
        roi_y2 = max(0, min(y + rh, img_h))
        minimap = img[roi_y1:roi_y2, roi_x1:roi_x2]

        # 阶段三：将 ROI 居中放置于 128×128 黑色画布
        size = CONFIG["OUTPUT_SIZE"]
        canvas = np.zeros((size, size, 3), dtype=np.uint8)
        cur_h, cur_w = minimap.shape[:2]
        if cur_h == 0 or cur_w == 0:
            return canvas

        dst_x = max(0, (size - cur_w) // 2)
        dst_y = max(0, (size - cur_h) // 2)
        src_x = max(0, (cur_w - size) // 2)
        src_y = max(0, (cur_h - size) // 2)
        copy_w = min(cur_w - src_x, size - dst_x)
        copy_h = min(cur_h - src_y, size - dst_y)

        canvas[dst_y : dst_y + copy_h, dst_x : dst_x + copy_w] = minimap[
            src_y : src_y + copy_h, src_x : src_x + copy_w
        ]

        # 阶段四：圆形 Mask，消除小地图外框噪声
        mask = np.zeros((size, size), dtype=np.uint8)
        cv2.circle(mask, (size // 2, size // 2), CONFIG["MASK_DIAMETER"] // 2, 255, -1)

        return cv2.bitwise_and(canvas, canvas, mask=mask)

    def predict(
        self,
        image_path: str,
        save_debug: bool = False,
        debug_path: str = DEFAULT_OPTIONS["debug_path"],
    ) -> list[tuple[str, float]]:
        """对单张图片执行推理，返回按置信度降序排列的类别结果。

        Args:
            image_path: 输入图片路径。
            save_debug: 为 True 时将预处理结果保存为 debug_inference.jpg。
            debug_path: 预处理结果的保存路径。

        Returns:
            列表，每项为 (class_name, confidence) 元组，按置信度降序排列。

        Raises:
            FileNotFoundError: 图片文件不存在时抛出。
            ValueError:        图片无法被 OpenCV 读取时抛出。
        """
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")

        img = safe_imread(path)
        if img is None:
            raise ValueError(f"Failed to read image: {path}")

        processed = self.preprocess(img)

        if save_debug:
            cv2.imwrite(debug_path, processed)
            logger.info(f"Debug image saved to: {debug_path}")

        results = self.model(processed, verbose=False)

        probs = results[0].probs.data.tolist()
        names = results[0].names
        all_results = [(names[i], conf) for i, conf in enumerate(probs)]
        all_results.sort(key=lambda x: x[1], reverse=True)
        return all_results


def parse_args() -> argparse.Namespace:
    """解析推理输入和命令行覆盖项。"""
    parser = argparse.ArgumentParser(description="YOLO Classification Inference Script")
    parser.add_argument("image", help="Path to the input image")
    parser.add_argument(
        "--model",
        default=argparse.SUPPRESS,
        help="Path to model weights (optional, defaults to latest training run)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Save the preprocessed image as debug_inference.jpg",
    )
    parser.add_argument(
        "--debug-path",
        default=argparse.SUPPRESS,
        help=f"Path for debug image output (default: {DEFAULT_OPTIONS['debug_path']})",
    )

    options = DEFAULT_OPTIONS.copy()
    options.update(vars(parser.parse_args()))
    return argparse.Namespace(**options)


if __name__ == "__main__":
    args = parse_args()
    try:
        engine = Predictor(args.model)
        predictions = engine.predict(args.image, args.debug, args.debug_path)
        print("\n>>> Predictions:")
        for name, conf in predictions:
            print(f"  {name}: {conf:.2%}")
        print()
    except Exception as e:
        logger.error(f"Error: {e}")
