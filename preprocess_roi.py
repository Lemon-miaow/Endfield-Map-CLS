"""
preprocess_roi.py — 小地图 ROI 预处理脚本

从原始截图中裁取固定坐标的小地图区域，将其居中放置于 128x128 画布，
并应用圆形 Mask 遮罩，用于将游戏截图转化为与 C++ 推理端逻辑等价的输入格式。

用法:
    python preprocess_roi.py -i <输入路径> [-o <输出目录>]

参数:
    -i / --input   输入图片路径，或包含图片的目录（支持子目录递归）
    -o / --output  处理结果保存目录，默认为 processed_output
"""

import argparse
import logging
from pathlib import Path

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# 小地图在原始截图中的 ROI 坐标（与 C++ 推理端保持一致）
_ROI_X, _ROI_Y, _ROI_W, _ROI_H = 49, 51, 118, 120

# 输出规格（与训练集 / C++ 推理端保持一致）
_OUTPUT_SIZE = 128
_MASK_DIAMETER = 106


def process_image(img: np.ndarray) -> np.ndarray:
    """从原始截图中提取并规范化小地图区域。

    处理流程：
        1. 按固定坐标裁取 ROI，并做边界钳位防止越界。
        2. 统一转换为 BGR 三通道（兼容 BGRA 和灰度输入）。
        3. 将 ROI 居中放置于 128x128 黑色画布。
        4. 应用半径为 53px 的圆形 Mask，消除方形边角噪声。

    Args:
        img: 原始截图，支持 BGR、BGRA 或灰度格式。

    Returns:
        处理后的 128x128 BGR 图像。若 ROI 区域为空则返回全黑图像。
    """
    img_h, img_w = img.shape[:2]

    # 坐标钳位，防止 ROI 越出图像边界
    roi_y1 = max(0, min(_ROI_Y, img_h))
    roi_y2 = max(0, min(_ROI_Y + _ROI_H, img_h))
    roi_x1 = max(0, min(_ROI_X, img_w))
    roi_x2 = max(0, min(_ROI_X + _ROI_W, img_w))

    minimap = img[roi_y1:roi_y2, roi_x1:roi_x2]

    canvas = np.zeros((_OUTPUT_SIZE, _OUTPUT_SIZE, 3), dtype=np.uint8)

    # 统一转换为 BGR 三通道
    if len(minimap.shape) == 3 and minimap.shape[2] == 4:
        img3c = cv2.cvtColor(minimap, cv2.COLOR_BGRA2BGR)
    elif len(minimap.shape) == 2:
        img3c = cv2.cvtColor(minimap, cv2.COLOR_GRAY2BGR)
    else:
        img3c = minimap.copy()

    cur_h, cur_w = img3c.shape[:2]
    if cur_h == 0 or cur_w == 0:
        return canvas

    # 计算居中偏移与裁剪范围，保证 ROI 超出 OUTPUT_SIZE 时也能安全放置
    start_y = max(0, (_OUTPUT_SIZE - cur_h) // 2)
    start_x = max(0, (_OUTPUT_SIZE - cur_w) // 2)
    crop_h = min(cur_h, _OUTPUT_SIZE)
    crop_w = min(cur_w, _OUTPUT_SIZE)
    img_roi_y = max(0, (cur_h - crop_h) // 2)
    img_roi_x = max(0, (cur_w - crop_w) // 2)

    canvas[start_y : start_y + crop_h, start_x : start_x + crop_w] = img3c[
        img_roi_y : img_roi_y + crop_h, img_roi_x : img_roi_x + crop_w
    ]

    # 圆形 Mask：消除 128x128 方形画布四角的无效区域
    mask = np.zeros((_OUTPUT_SIZE, _OUTPUT_SIZE), dtype=np.uint8)
    cv2.circle(mask, (_OUTPUT_SIZE // 2, _OUTPUT_SIZE // 2), _MASK_DIAMETER // 2, 255, -1)

    return cv2.bitwise_and(canvas, canvas, mask=mask)


def safe_imread(path: Path) -> np.ndarray:
    """读取图片，兼容路径中包含非 ASCII 字符（如中文）的情况。"""
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_UNCHANGED)


def safe_imwrite(path: Path, img: np.ndarray) -> bool:
    """写入图片，兼容路径中包含非 ASCII 字符（如中文）的情况。"""
    ext = path.suffix
    is_success, buf = cv2.imencode(ext, img)
    if is_success:
        buf.tofile(str(path))
    return is_success


def batch_process(input_path: str, output_dir: str) -> None:
    """批量处理输入路径下的所有图片。

    若输入为单张图片，则仅处理该文件；
    若输入为目录，则递归查找所有 PNG / JPG 文件（忽略大小写）。
    输出目录结构与输入保持一致。

    Args:
        input_path: 输入图片路径或包含图片的目录路径。
        output_dir:  处理结果保存目录。
    """
    in_path = Path(input_path)
    out_dir = Path(output_dir)

    if not in_path.exists():
        logger.error(f"输入路径不存在: {in_path}")
        return

    files_to_process: list[Path] = []
    if in_path.is_file():
        files_to_process.append(in_path)
    else:
        for ext in ["*.[pP][nN][gG]", "*.[jJ][pP][gG]"]:
            files_to_process.extend(in_path.rglob(ext))

    if not files_to_process:
        logger.warning(f"在 {in_path} 中未找到图片文件")
        return

    logger.info(f"找到 {len(files_to_process)} 张图片，开始处理...")

    for file_path in files_to_process:
        try:
            img = safe_imread(file_path)
            if img is None:
                logger.warning(f"无法读取图片，跳过: {file_path}")
                continue

            processed = process_image(img)

            # 保持相对目录结构写入输出目录
            rel_path = file_path.name if in_path.is_file() else file_path.relative_to(in_path)
            out_file_path = out_dir / rel_path
            out_file_path.parent.mkdir(parents=True, exist_ok=True)

            if safe_imwrite(out_file_path, processed):
                logger.info(f"成功: {file_path.name}")
            else:
                logger.error(f"保存失败: {out_file_path}")

        except Exception as e:
            logger.error(f"处理图片 {file_path} 时出错: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="小地图 ROI 批量预处理脚本（与 C++ 推理端逻辑一致）"
    )
    parser.add_argument(
        "-i", "--input", required=True, help="输入图片路径 或 包含图片的目录"
    )
    parser.add_argument(
        "-o",
        "--output",
        default="processed_output",
        help="处理后图片保存的目录（默认: processed_output）",
    )
    args = parser.parse_args()

    batch_process(args.input, args.output)
