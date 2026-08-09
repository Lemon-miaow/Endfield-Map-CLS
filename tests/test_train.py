from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from ultralytics.models.yolo.classify import ClassificationValidator

from train import ValidationLossValidator, find_latest_model


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
        self.assertIn("target=90.00%", logs.output[0])

    def test_fixed_loss_has_full_weight_in_checkpoint_fitness(self) -> None:
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

        self.assertAlmostEqual(metrics["fitness"], 1.0 / 1.6)
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
