from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from fine_tune import fine_tune, select_better_checkpoint
from train import ValidationLossTrainer


class FineTuneTests(unittest.TestCase):
    def test_fine_tune_uses_latest_best_with_low_learning_rate(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "best.pt"
            candidate = root / "finetune" / "weights" / "best.pt"
            selected = candidate.with_name("selected.pt")
            source.touch()
            candidate.parent.mkdir(parents=True)
            candidate.touch()

            model = Mock()
            model.trainer = SimpleNamespace(best=candidate)
            with (
                patch("fine_tune.find_latest_model", return_value=str(source)),
                patch("fine_tune.YOLO", return_value=model) as yolo,
                patch(
                    "fine_tune.select_better_checkpoint",
                    return_value=selected,
                ) as select,
            ):
                result = fine_tune()

            yolo.assert_called_once_with(str(source.resolve()))
            options = model.train.call_args.kwargs
            self.assertIs(options["trainer"], ValidationLossTrainer)
            self.assertEqual(options["optimizer"], "MuSGD")
            self.assertEqual(options["lr0"], 0.001)
            self.assertEqual(options["lrf"], 0.1)
            self.assertEqual(options["warmup_epochs"], 0.0)
            self.assertEqual(options["epochs"], 15)
            self.assertEqual(options["patience"], 5)
            select.assert_called_once_with(source.resolve(), candidate)
            self.assertEqual(result, selected)

    def test_selection_keeps_the_checkpoint_with_higher_fitness(self) -> None:
        cases = [
            (0.8, 0.7, b"coarse"),
            (0.8, 0.9, b"fine"),
        ]
        for source_fitness, candidate_fitness, expected in cases:
            with self.subTest(candidate_fitness=candidate_fitness):
                with tempfile.TemporaryDirectory() as temp_dir:
                    root = Path(temp_dir)
                    source = root / "coarse.pt"
                    candidate = root / "best.pt"
                    source.write_bytes(b"coarse")
                    candidate.write_bytes(b"fine")
                    os.utime(source, (1, 1))
                    os.utime(candidate, (1, 1))

                    with patch(
                        "fine_tune.checkpoint_fitness",
                        side_effect=[source_fitness, candidate_fitness],
                    ):
                        selected = select_better_checkpoint(source, candidate)

                    self.assertEqual(selected.read_bytes(), expected)
                    self.assertGreater(selected.stat().st_mtime, 1)


if __name__ == "__main__":
    unittest.main()
