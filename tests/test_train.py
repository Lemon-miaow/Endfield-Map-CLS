from __future__ import annotations

import math
import os
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import cv2
import numpy as np
import torch
from PIL import Image
from ultralytics.cfg import get_cfg
from ultralytics.models.yolo.classify import ClassificationValidator

from preprocess import CONFIG as PREPROCESS_CONFIG
from train import (
    FIXED_WORST_LOSS_WEIGHT,
    ONLINE_ZOOM_AREA,
    CenterZoom,
    ValidationLossTrainer,
    ValidationLossValidator,
    find_latest_model,
    parse_args,
    train,
)


class TrainingGeometryTests(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name) / "dataset"
        class_dir = self.root / "map"
        class_dir.mkdir(parents=True)
        yy, xx = np.indices((128, 128))
        pixels = np.stack((xx * 2, yy * 2, (xx + yy) % 255), axis=-1).astype(np.uint8)
        pixels[60:68, 62:66] = (255, 255, 255)
        circle = np.zeros((128, 128), dtype=np.uint8)
        cv2.circle(circle, (64, 64), 53, 255, -1)
        pixels[circle == 0] = 0
        Image.fromarray(pixels).save(class_dir / "sample.png")
        self.expected = torch.from_numpy(pixels).permute(2, 0, 1).float() / 255.0
        self.outside = torch.from_numpy(circle == 0)
        self.trainer = object.__new__(ValidationLossTrainer)
        self.trainer.args = get_cfg(
            overrides={
                "imgsz": 128,
                "hsv_h": 0.0,
                "hsv_s": 0.0,
                "hsv_v": 0.0,
                "auto_augment": None,
                "erasing": 0.0,
            }
        )

    def test_training_zooms_around_player_inside_minimap_mask(self) -> None:
        dataset = self.trainer.build_dataset(str(self.root))

        images = [dataset[0]["img"] for _ in range(64)]
        for image in images:
            self.assertEqual(image.shape, self.expected.shape)
            # 玩家白块随 ±6px 平移和镜像移动，但始终留在中心附近。
            self.assertTrue(torch.all(image[:, 54:74, 54:74].amax(dim=(1, 2)) > 0.95))
            self.assertTrue(torch.all(image[:, self.outside] == 0))
        self.assertGreater(len({image.numpy().tobytes() for image in images}), 1)

    def test_center_zoom_scales_offsets_from_player_pixel(self) -> None:
        pixels = np.zeros((128, 128, 3), dtype=np.uint8)
        pixels[64, 64] = 255
        pixels[64, 84] = 255

        zoomed = CenterZoom().apply(pixels, 1.2)

        self.assertEqual(int(zoomed[64, 64, 0]), 255)
        self.assertEqual(int(np.argmax(zoomed[64, 70:, 0])) + 70, 88)

    def test_center_zoom_shift_and_flip_move_player_pixel(self) -> None:
        pixels = np.zeros((128, 128, 3), dtype=np.uint8)
        pixels[64, 64] = 255
        pixels[64, 84] = 255

        shifted = CenterZoom().apply(pixels, 1.0, dx=3, dy=-2)
        self.assertEqual(int(shifted[62, 67, 0]), 255)
        self.assertEqual(int(shifted[62, 87, 0]), 255)

        flipped = CenterZoom().apply(pixels, 1.0, flip=True)
        self.assertEqual(int(flipped[64, 63, 0]), 255)
        self.assertEqual(int(flipped[64, 43, 0]), 255)

        random.seed(0)
        outputs = {CenterZoom()(Image.fromarray(pixels)).tobytes() for _ in range(16)}
        self.assertGreater(len(outputs), 8)

    def test_online_zoom_is_the_only_scale_source(self) -> None:
        # 小地图尺度固定，预处理不做缩放；训练加载的中心放大最大 1/√0.7。
        self.assertFalse([key for key in PREPROCESS_CONFIG if key.startswith("SCALE_JITTER")])
        self.assertAlmostEqual(1 / math.sqrt(ONLINE_ZOOM_AREA[0]), 1.195, delta=0.001)

    def test_cuda_graph_compile_is_default_and_can_be_disabled(self) -> None:
        with patch("sys.argv", ["train.py"]):
            self.assertEqual(parse_args().compile, "reduce-overhead")
        with patch("sys.argv", ["train.py", "--compile", "False"]):
            self.assertIs(parse_args().compile, False)

    def test_shutdown_flag_is_off_by_default_and_runs_after_training(self) -> None:
        with patch("sys.argv", ["train.py"]):
            self.assertFalse(getattr(parse_args(), "shutdown", False))
        with patch("sys.argv", ["train.py", "--shutdown"]):
            self.assertTrue(parse_args().shutdown)

        args = SimpleNamespace(model="yolo26s-cls.pt", data="d", epochs=1, imgsz=128, batch=1, nbs=1, workers=0,
                               device="cpu", patience=1, erasing=0.0, auto_augment=None, compile=False,
                               project="p", name="n", shutdown=True)
        with patch("train.YOLO") as yolo, patch("train.shutdown_machine") as shutdown:
            yolo.return_value.train.side_effect = RuntimeError("boom")
            with self.assertRaises(RuntimeError):
                train(args)
        shutdown.assert_called_once()

    def test_color_augmentation_still_varies(self) -> None:
        self.trainer.args.hsv_h = 0.015
        self.trainer.args.hsv_s = 0.7
        self.trainer.args.hsv_v = 0.4
        dataset = self.trainer.build_dataset(str(self.root))

        first = dataset[0]["img"]
        second = dataset[0]["img"]
        self.assertFalse(torch.equal(first, second))
        self.assertEqual(first.shape, self.expected.shape)

    def test_validation_remains_unaugmented(self) -> None:
        self.trainer.args.hsv_v = 0.4
        dataset = self.trainer.build_dataset(str(self.root), mode="val")

        self.assertTrue(torch.equal(dataset[0]["img"], self.expected))


class ValidationSelectionTests(unittest.TestCase):
    def test_latest_model_includes_selected_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            best = root / "train" / "weights" / "best.pt"
            selected = root / "finetune" / "weights" / "selected.pt"
            best.parent.mkdir(parents=True)
            selected.parent.mkdir(parents=True)
            best.touch()
            selected.touch()
            os.utime(best, (1, 1))
            os.utime(selected, (2, 2))

            self.assertEqual(find_latest_model(str(root)), str(selected))

            os.utime(best, (3, 3))
            self.assertEqual(find_latest_model(str(root)), str(best))

            os.utime(selected, (3, 3))
            self.assertEqual(find_latest_model(str(root)), str(selected))

    def test_fixed_evaluation_uses_logits_and_existing_probabilities(self) -> None:
        validator = object.__new__(ValidationLossValidator)
        dataset = Mock()
        dataset.samples = [("/tmp/fixed_example.png", 1)]
        dataset.__getitem__ = Mock(
            return_value={"img": torch.zeros((3, 128, 128))}
        )
        validator.dataloader = SimpleNamespace(dataset=dataset)
        validator.device = torch.device("cpu")
        validator.names = {0: "wrong", 1: "right"}

        probabilities = torch.tensor([[0.1, 0.9]])
        logits = torch.tensor([[-2.0, 2.0]])
        model = Mock(return_value=(probabilities, logits))
        trainer = SimpleNamespace(
            rank=-1,
            ema=SimpleNamespace(ema=model),
            model=None,
            args=SimpleNamespace(compile=False),
            epoch=0,
            epochs=1,
        )

        with self.assertLogs("train", level="INFO") as logs:
            metrics = validator._evaluate_fixed_predictions(trainer)

        expected_loss = torch.nn.functional.cross_entropy(
            logits,
            torch.tensor([1]),
        )
        self.assertAlmostEqual(metrics["loss"], float(expected_loss))
        self.assertEqual(metrics["top1_acc"], 1.0)
        output = "\n".join(logs.output)
        self.assertIn("[Fixed Val][1/1] 1/1 OK", output)
        self.assertIn("TOP2", output)
        self.assertIn("90.00%", output)
        self.assertIn("wrong", output)
        self.assertIn("80.00pp", output)

    def test_fixed_mean_and_worst_loss_select_checkpoint(self) -> None:
        validator = object.__new__(ValidationLossValidator)
        validator._evaluate_fixed_predictions = Mock(
            return_value={"loss": 0.4, "worst_loss": 0.8, "top1_acc": 0.75}
        )

        with patch.object(
            ClassificationValidator,
            "__call__",
            return_value={"val/loss": 0.2},
        ):
            metrics = validator(trainer=object())

        expected = 1.0 / (1.0 + 0.2 + 0.4 + FIXED_WORST_LOSS_WEIGHT * 0.8)
        self.assertAlmostEqual(metrics["fitness"], expected)
        self.assertEqual(metrics["fixed_val/loss"], 0.4)
        self.assertEqual(metrics["fixed_val/worst_loss"], 0.8)
        self.assertEqual(metrics["fixed_val/top1_acc"], 0.75)

    def test_generated_loss_remains_selection_fallback_without_fixed_val(self) -> None:
        validator = object.__new__(ValidationLossValidator)
        validator._evaluate_fixed_predictions = Mock(return_value=None)

        with patch.object(
            ClassificationValidator,
            "__call__",
            return_value={"val/loss": 0.2},
        ):
            metrics = validator(trainer=object())

        self.assertAlmostEqual(metrics["fitness"], 1.0 / 1.2)
        self.assertNotIn("fixed_val/loss", metrics)


if __name__ == "__main__":
    unittest.main()
