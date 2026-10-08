from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from report import INK, changed_rows, delta_text, source_label


class ReportLabelTests(unittest.TestCase):
    def test_source_labels_read_file_names(self) -> None:
        self.assertEqual(source_label(Path("issue_5571_cave_facilities.png")), "issue #5571")
        self.assertEqual(source_label(Path("autocollect_route4_loc.png")), "自动采集定位帧")
        self.assertEqual(source_label(Path("testset_transfer_87_white_flowers_close.png")), "测试集·白花近景")
        self.assertEqual(source_label(Path("testset_port_storager_6_dark_forest.png")), "测试集·暗林")
        self.assertEqual(source_label(Path("619524064-84a6f303.png")), "issue 附图")
        self.assertEqual(source_label(Path("Base02_r03_c07.png")), "实机帧")

    def test_changed_rows_keep_moves_and_low_scores_sorted_by_gain(self) -> None:
        old = np.array([99.9, 8.7, 94.5, 99.5, 90.0])
        new = np.array([99.95, 97.0, 87.9, 99.9, 90.1])
        self.assertEqual(changed_rows(old, new), [1, 4, 2])

    def test_delta_arrow_follows_value_and_color_follows_goodness(self) -> None:
        self.assertEqual(delta_text(1.81, 2), ("▲ 1.81", INK["up"]))
        self.assertEqual(delta_text(-6, 0, lower_is_better=True), ("▼ 6", INK["up"]))
        self.assertEqual(delta_text(-6.6), ("▼ 6.6", INK["down"]))
        self.assertEqual(delta_text(0.01), ("持平", INK["muted"]))


if __name__ == "__main__":
    unittest.main()
