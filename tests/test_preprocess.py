from __future__ import annotations

import math
import random
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from preprocess import (
    BackgroundProfile,
    CONFIG,
    UI_BLUE_BGR,
    UiClutter,
    ZONE_BLUE_BGR,
    ZONE_YELLOW_BGR,
    _sample_normal_ui_icon,
    add_black_outline_rgba,
    add_error_training_samples,
    add_extreme_icon_clutter,
    add_tier_center_icon_cluster,
    apply_background_composition,
    build_balanced_schedule,
    build_scale_jitter_schedule,
    build_tier_center_mask,
    build_tier_center_ui_plan,
    build_ui_clutter_schedule,
    build_zone_plan,
    build_zone_ui_clutter_schedule,
    copy_fixed_validation_samples,
    draw_random_ui_lines,
    finalize_positive_map_sample,
    generate_samples,
    is_valid,
    load_ui_icons,
    safe_imwrite,
    sample_minimap_center,
    save_dataset,
    tint_icon_blue_rgba,
)


class BalancedSamplingTests(unittest.TestCase):
    def test_full_cycles_cover_every_center_equally(self) -> None:
        random.seed(7)
        centers = list(range(400))

        schedule = build_balanced_schedule(centers, 1200)
        counts = Counter(schedule)

        self.assertEqual(len(schedule), 1200)
        self.assertEqual(set(counts), set(centers))
        self.assertEqual(set(counts.values()), {3})

    def test_partial_cycle_differs_by_at_most_one(self) -> None:
        random.seed(7)
        schedule = build_balanced_schedule(list(range(7)), 17)

        counts = Counter(schedule)

        self.assertEqual(len(counts), 7)
        self.assertLessEqual(max(counts.values()) - min(counts.values()), 1)

    def test_zone_plan_guarantees_yellow_coverage_and_spreads_blue(self) -> None:
        random.seed(7)
        centers = [(index, 0) for index in range(400)]

        train_plan, split_plan = build_zone_plan(centers, 600)
        plan = train_plan + split_plan
        yellow_centers = [center for center, color in plan if color == ZONE_YELLOW_BGR]
        blue_centers = [center for center, color in plan if color == ZONE_BLUE_BGR]

        self.assertEqual(len(train_plan), 400)
        self.assertEqual({center for center, _color in train_plan}, set(centers))
        self.assertEqual(len(yellow_centers), 450)
        self.assertEqual(len(blue_centers), 150)
        self.assertEqual(set(yellow_centers), set(centers))
        self.assertEqual(len(set(blue_centers)), 150)

    def test_ui_clutter_schedule_keeps_ratio_and_reserves_ultra_cases(self) -> None:
        random.seed(7)

        schedule = build_ui_clutter_schedule(1200)

        self.assertEqual(len(schedule), 1200)
        self.assertEqual(schedule.count(UiClutter.NONE), 1139)
        self.assertEqual(schedule.count(UiClutter.EXTREME), 51)
        self.assertEqual(schedule.count(UiClutter.ULTRA), 10)

        tier_schedule = build_ui_clutter_schedule(300)
        self.assertEqual(tier_schedule.count(UiClutter.NONE), 285)
        self.assertEqual(tier_schedule.count(UiClutter.EXTREME), 12)
        self.assertEqual(tier_schedule.count(UiClutter.ULTRA), 3)

    def test_zone_ui_clutter_schedule_covers_yellow_and_blue(self) -> None:
        random.seed(7)
        plan = [((index, 0), ZONE_YELLOW_BGR) for index in range(50)]
        plan.extend(
            ((index, 1), ZONE_BLUE_BGR)
            for index in range(150)
        )

        schedule = build_zone_ui_clutter_schedule(plan)
        yellow = schedule[:50]
        blue = schedule[50:]

        self.assertEqual(sum(level != UiClutter.NONE for level in yellow), 3)
        self.assertEqual(sum(level != UiClutter.NONE for level in blue), 8)
        self.assertEqual(yellow.count(UiClutter.ULTRA), 1)
        self.assertEqual(blue.count(UiClutter.ULTRA), 1)

    def test_scale_jitter_schedule_is_balanced_and_keeps_clean_majority(self) -> None:
        random.seed(7)

        schedule = build_scale_jitter_schedule(1200)
        smaller = [scale for scale in schedule if scale < 1.0]
        larger = [scale for scale in schedule if scale > 1.0]

        self.assertEqual(schedule.count(1.0), 900)
        self.assertEqual(len(smaller), 150)
        self.assertEqual(len(larger), 150)
        self.assertGreaterEqual(min(smaller), CONFIG["SCALE_JITTER_MIN"])
        self.assertLessEqual(max(larger), CONFIG["SCALE_JITTER_MAX"])


class UiCompositionTests(unittest.TestCase):
    def test_icon_outline_is_soft_not_opaque(self) -> None:
        icon = np.zeros((5, 5, 4), dtype=np.uint8)
        icon[2, 2] = (255, 255, 255, 255)

        outlined = add_black_outline_rgba(icon)
        outline_alpha = int(outlined[2, 1, 3])

        self.assertGreater(outline_alpha, 0)
        self.assertLess(outline_alpha, 255)

    def test_route_lines_are_antialiased_and_translucent(self) -> None:
        random.seed(1)
        image = np.zeros((128, 128, 3), dtype=np.uint8)

        result = draw_random_ui_lines(image)

        values = np.unique(result)
        self.assertGreater(values.size, 4)
        self.assertLess(int(result.max()), 255)

    def test_blue_tint_keeps_internal_dark_details(self) -> None:
        icon = np.zeros((3, 3, 4), dtype=np.uint8)
        icon[..., 3] = 255
        icon[0, 0, :3] = 255

        tinted = tint_icon_blue_rgba(icon)

        self.assertTupleEqual(tuple(tinted[0, 0, :3]), UI_BLUE_BGR)
        self.assertTupleEqual(tuple(tinted[1, 1, :3]), (0, 0, 0))

    def test_icon_scale_and_outline_match_real_minimap_range(self) -> None:
        icons = load_ui_icons("icon")
        name = "icon_port_power_pole_2.png"

        with (
            patch("preprocess.random.uniform", return_value=1.0),
            patch("preprocess.random.random", return_value=0.0),
        ):
            icon = _sample_normal_ui_icon(icons["normal"], [name], name)

        self.assertGreaterEqual(min(icon.shape[:2]), 10)
        self.assertLessEqual(max(icon.shape[:2]), 20)
        solid = icon[..., 3] > 32
        self.assertGreater(np.count_nonzero(np.all(icon[..., :3] == UI_BLUE_BGR, axis=2)), 0)
        self.assertGreater(np.count_nonzero(solid & np.all(icon[..., :3] == 0, axis=2)), 0)

    def test_icon_centers_stay_inside_minimap_circle(self) -> None:
        random.seed(7)
        center = CONFIG["OUTPUT_SIZE"] / 2
        max_radius = CONFIG["MASK_DIAMETER"] / 2 - 3

        for edge_bias in (False, True):
            for _ in range(200):
                x, y = sample_minimap_center(
                    CONFIG["OUTPUT_SIZE"],
                    CONFIG["OUTPUT_SIZE"],
                    edge_bias=edge_bias,
                )
                self.assertLessEqual(math.hypot(x - center, y - center), max_radius + 1)

    def test_ultra_icon_count_is_total_budget_not_per_pack(self) -> None:
        image = np.zeros((128, 128, 3), dtype=np.uint8)
        normal_icons = {"icon.png": np.zeros((8, 8, 4), dtype=np.uint8)}
        sampled = np.full((1, 1, 4), 255, dtype=np.uint8)

        with (
            patch.dict(
                CONFIG,
                {
                    "ULTRA_UI_PACK_MIN": 4,
                    "ULTRA_UI_PACK_MAX": 4,
                    "ULTRA_UI_ICONS_MIN": 28,
                    "ULTRA_UI_ICONS_MAX": 28,
                    "ULTRA_UI_CHAIN_PROB": 0.0,
                },
            ),
            patch("preprocess._sample_normal_ui_icon", return_value=sampled),
            patch("preprocess.draw_one_normal_icon", side_effect=lambda result, *_: result) as draw,
        ):
            add_extreme_icon_clutter(
                image,
                normal_icons,
                ["icon.png"],
                UiClutter.ULTRA,
            )

        self.assertEqual(draw.call_count, 28)

    def test_real_landmark_asset_is_loaded_separately(self) -> None:
        icons = load_ui_icons("icon")

        self.assertEqual(len(icons["landmarks"]), 1)
        self.assertTupleEqual(icons["landmarks"][0].shape[:2], (22, 22))

    def test_tier_center_cluster_combines_landmark_and_large_icons(self) -> None:
        image = np.zeros((128, 128, 3), dtype=np.uint8)
        normal_icons = {"icon.png": np.full((8, 8, 4), 255, dtype=np.uint8)}
        landmarks = [np.full((12, 12, 4), 255, dtype=np.uint8)]
        sampled = np.full((4, 4, 4), 255, dtype=np.uint8)

        with (
            patch.dict(
                CONFIG,
                {
                    "TIER_CENTER_UI_ICONS_MIN": 6,
                    "TIER_CENTER_UI_ICONS_MAX": 6,
                    "TIER_CENTER_UI_RADIUS": 0,
                },
            ),
            patch("preprocess._sample_normal_ui_icon", return_value=sampled),
            patch(
                "preprocess.overlay_rgba_on_bgr",
                side_effect=lambda result, *_: result,
            ) as overlay,
        ):
            add_tier_center_icon_cluster(
                image,
                normal_icons,
                ["icon.png"],
                landmarks,
            )

        self.assertEqual(overlay.call_count, 6)

    def test_positive_sample_always_contains_center_pointer(self) -> None:
        random.seed(7)
        image = np.zeros(
            (CONFIG["OUTPUT_SIZE"], CONFIG["OUTPUT_SIZE"], 3),
            dtype=np.uint8,
        )

        result = finalize_positive_map_sample(image)
        center = CONFIG["OUTPUT_SIZE"] // 2
        pointer_region = result[center - 20 : center + 20, center - 20 : center + 20]

        self.assertGreater(np.count_nonzero(pointer_region), 0)

    def test_base_keeps_one_clean_anchor_pass_before_ui_augmentation(self) -> None:
        safe_size = 182
        pad = safe_size // 2
        size = 256
        yy, xx = np.indices((size, size))
        texture = np.stack(
            (
                (xx * 3 + yy) % 255,
                (xx + yy * 5) % 255,
                (xx * 7 + yy * 2) % 255,
            ),
            axis=2,
        ).astype(np.uint8)
        image = np.zeros((size + pad * 2, size + pad * 2, 4), dtype=np.uint8)
        image[pad : pad + size, pad : pad + size, :3] = texture
        image[pad : pad + size, pad : pad + size, 3] = 255

        with patch("preprocess.augment_patch", side_effect=lambda patch, *_: patch) as augment:
            generate_samples(
                image,
                safe_size,
                [],
                sample_region=(96, 96, 64, 64),
                target_count=96,
            )

        self.assertEqual(augment.call_count, 32)


class TierSamplingTests(unittest.TestCase):
    def test_center_ui_plan_covers_every_center_and_balances_zones(self) -> None:
        random.seed(7)
        centers = [(index, 0) for index in range(100)]

        plan = build_tier_center_ui_plan(centers, target_count=20)
        colors = [color for _center, color in plan]

        self.assertEqual(len(plan), 100)
        self.assertEqual({center for center, _color in plan}, set(centers))
        self.assertEqual(colors.count(ZONE_YELLOW_BGR), 49)
        self.assertEqual(colors.count(ZONE_BLUE_BGR), 16)
        self.assertEqual(colors.count(None), 35)

    def test_center_mask_keeps_only_structure_neighborhood(self) -> None:
        image = np.zeros((64, 64, 4), dtype=np.uint8)
        image[:8, :8, 3] = 255
        image[28:36, 28:36] = (255, 255, 255, 255)

        mask = build_tier_center_mask(image)

        self.assertTrue(mask[32, 32])
        self.assertTrue(mask[32, 24])
        self.assertFalse(mask[4, 4])
        self.assertFalse(mask[0, 63])

    def test_small_centered_tier_uses_tier_coverage_thresholds(self) -> None:
        image = np.zeros((128, 128, 4), dtype=np.uint8)
        image[47:81, 47:81, 3] = 255
        image[56:72, 56:64, :3] = 100
        image[56:72, 64:72, :3] = 255

        self.assertFalse(is_valid(image))
        min_map = CONFIG["TIER_MIN_MAP_CIRCLE_COVERAGE"]
        min_alpha = CONFIG["TIER_MIN_ALPHA_CIRCLE_COVERAGE"]
        self.assertTrue(
            is_valid(
                image,
                min_map_circle_coverage=min_map,
                min_alpha_circle_coverage=min_alpha,
            )
        )

    def test_positive_map_background_is_always_black(self) -> None:
        image = np.zeros((32, 32, 4), dtype=np.uint8)
        image[8:24, 8:24] = (30, 60, 90, 255)

        standard = apply_background_composition(image, [Path("unused.png")])
        tier = apply_background_composition(image, [], BackgroundProfile.TIER)

        for result in (standard, tier):
            self.assertTupleEqual(tuple(result[0, 0]), (0, 0, 0))
            self.assertTupleEqual(tuple(result[16, 16]), (30, 60, 90))

    def test_low_signal_base_centers_never_receive_ui_or_zones(self) -> None:
        safe_size = 182
        pad = safe_size // 2
        image = np.zeros((256 + pad * 2, 256 + pad * 2, 4), dtype=np.uint8)
        image[pad : pad + 256, pad : pad + 256, 3] = 255

        def valid_for_clean_anchor(_image, **kwargs):
            return kwargs.get("min_map_center_coverage") is None

        with (
            patch("preprocess.is_valid", side_effect=valid_for_clean_anchor),
            patch("preprocess.augment_patch") as augment,
            patch("preprocess.augment_zone_patch") as augment_zone,
            patch(
                "preprocess.finalize_positive_map_sample",
                side_effect=lambda sample: sample,
            ),
        ):
            samples, train_only_samples = generate_samples(
                image,
                safe_size,
                [],
                sample_region=(96, 96, 64, 64),
                target_count=96,
            )

        self.assertEqual(len(samples), 32)
        self.assertEqual(len(train_only_samples), 64)
        augment.assert_not_called()
        augment_zone.assert_not_called()

    def test_base_pointer_center_never_lands_on_transparent_void(self) -> None:
        safe_size = 182
        pad = safe_size // 2
        image = np.zeros((256 + pad * 2, 256 + pad * 2, 4), dtype=np.uint8)
        image[pad : pad + 256, pad : pad + 128, 3] = 255
        sampled_centers = []

        def extract_at_center(_image, cx, cy, *_args):
            sampled_centers.append((cx - pad, cy - pad))
            patch = np.zeros((CONFIG["OUTPUT_SIZE"], CONFIG["OUTPUT_SIZE"], 4), dtype=np.uint8)
            patch[..., 3] = 255
            return patch

        with (
            patch("preprocess.extract_roi", side_effect=extract_at_center),
            patch("preprocess.is_valid", return_value=True),
            patch("preprocess.augment_patch", side_effect=lambda sample, *_: sample),
            patch(
                "preprocess.finalize_positive_map_sample",
                side_effect=lambda sample: sample,
            ),
            patch.dict(CONFIG, {"UI_ZONE_EXTRA_RATIO": 0.0}),
        ):
            generate_samples(
                image,
                safe_size,
                [],
                sample_region=(96, 96, 64, 64),
                target_count=64,
            )

        self.assertTrue(sampled_centers)
        self.assertTrue(all(x < 128 for x, _y in sampled_centers))

    def test_center_icon_samples_are_train_only_and_keep_base_quota(self) -> None:
        safe_size = 182
        pad = safe_size // 2
        size = 192
        yy, xx = np.indices((size, size))
        texture = np.stack(
            (
                (xx * 3 + yy) % 255,
                (xx + yy * 5) % 255,
                (xx * 7 + yy * 2) % 255,
            ),
            axis=2,
        ).astype(np.uint8)
        image = np.zeros((size + pad * 2, size + pad * 2, 4), dtype=np.uint8)
        image[pad : pad + size, pad : pad + size, :3] = texture
        image[pad : pad + size, pad : pad + size, 3] = 255
        tier_context = {
            "parent_aligned": np.zeros(image.shape[:2] + (3,), dtype=np.uint8),
            "mask_mode": "opaque",
        }

        with (
            patch.dict(
                CONFIG,
                {
                    "TIER_CENTER_UI_EXTRA_RATIO": 0.20,
                    "UI_ZONE_EXTRA_RATIO": 0.0,
                    "SCALE_JITTER_RATIO": 0.0,
                    "EXTREME_UI_PROB": 0.0,
                    "ULTRA_UI_PROB": 0.0,
                },
            ),
            patch(
                "preprocess.augment_patch_light",
                side_effect=lambda sample, clutter=UiClutter.NONE: np.full_like(
                    sample,
                    int(clutter),
                ),
            ),
            patch(
                "preprocess.finalize_positive_map_sample",
                side_effect=lambda sample: sample,
            ),
            patch(
                "preprocess.build_tier_center_ui_plan",
                return_value=[
                    ((pad + 64, pad + 64), None),
                    ((pad + 72, pad + 64), ZONE_YELLOW_BGR),
                    ((pad + 64, pad + 72), ZONE_BLUE_BGR),
                    ((pad + 72, pad + 72), None),
                ],
            ),
            patch(
                "preprocess.augment_zone_patch",
                side_effect=lambda sample, _color, clutter: np.full_like(
                    sample,
                    int(clutter),
                ),
            ),
        ):
            samples, train_only_samples = generate_samples(
                image,
                safe_size,
                [],
                target_count=20,
                light_aug=True,
                tier_context=tier_context,
            )

        self.assertEqual(len(samples), 20)
        self.assertEqual(len(train_only_samples), 4)
        self.assertTrue(
            all(
                np.all(sample == int(UiClutter.TIER_CENTER))
                for sample in train_only_samples
            )
        )


class CuratedSampleTests(unittest.TestCase):
    def test_protected_position_pairs_stay_out_of_random_validation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "dataset"
            split_samples = [np.zeros((8, 8, 3), dtype=np.uint8) for _ in range(8)]
            protected_samples = [
                np.full((8, 8, 3), 255, dtype=np.uint8) for _ in range(2)
            ]

            save_dataset(
                split_samples,
                "Map02Base__r16_c04",
                "sample",
                output_dir,
                protected_samples,
            )

            train_paths = list(
                (output_dir / "train" / "Map02Base__r16_c04").glob("*.jpg")
            )
            val_paths = list(
                (output_dir / "val" / "Map02Base__r16_c04").glob("*.jpg")
            )
            self.assertEqual(len(train_paths), 8)
            self.assertEqual(len(val_paths), 2)
            self.assertTrue(
                all(float(cv2.imread(str(path)).mean()) < 1 for path in val_paths)
            )

    def test_hard_samples_are_written_only_to_train_with_ratio_floor(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            error_dir = root / "error_images"
            output_dir = root / "dataset"
            class_name = "Map02Base__r16_c04"
            source_dir = error_dir / class_name
            train_dir = output_dir / "train" / class_name
            val_dir = output_dir / "val" / class_name
            source_dir.mkdir(parents=True)
            train_dir.mkdir(parents=True)
            val_dir.mkdir(parents=True)

            image = np.zeros(
                (CONFIG["OUTPUT_SIZE"], CONFIG["OUTPUT_SIZE"], 3),
                dtype=np.uint8,
            )
            safe_imwrite(source_dir / "failure.png", image)
            for index in range(96):
                (train_dir / f"generated_{index:03d}.jpg").touch()
            for index in range(24):
                (val_dir / f"generated_{index:03d}.jpg").touch()

            added = add_error_training_samples(error_dir, output_dir)

            self.assertEqual(added, 6)
            self.assertEqual(len(list(train_dir.glob("hard_*.jpg"))), 6)
            self.assertEqual(list(val_dir.glob("hard_*.jpg")), [])

    def test_fixed_validation_is_checked_and_copied(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            fixed_dir = root / "validation_images"
            output_dir = root / "dataset"
            class_name = "Map01Base__r04_c03"
            source_dir = fixed_dir / class_name
            (output_dir / "train" / class_name).mkdir(parents=True)
            (output_dir / "val" / class_name).mkdir(parents=True)
            source_dir.mkdir(parents=True)

            image = np.zeros(
                (CONFIG["OUTPUT_SIZE"], CONFIG["OUTPUT_SIZE"], 3),
                dtype=np.uint8,
            )
            safe_imwrite(source_dir / "regression.png", image)

            copied = copy_fixed_validation_samples(fixed_dir, output_dir)

            self.assertEqual(copied, 1)
            self.assertTrue(
                (output_dir / "val" / class_name / "fixed_regression.png").exists()
            )

    def test_fixed_validation_rejects_unprocessed_screenshot(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            fixed_dir = root / "validation_images"
            output_dir = root / "dataset"
            class_name = "Map01Base__r04_c03"
            source_dir = fixed_dir / class_name
            (output_dir / "train" / class_name).mkdir(parents=True)
            (output_dir / "val" / class_name).mkdir(parents=True)
            source_dir.mkdir(parents=True)
            safe_imwrite(
                source_dir / "full_screenshot.png",
                np.zeros((720, 1280, 3), dtype=np.uint8),
            )

            with self.assertRaisesRegex(ValueError, "must be 128x128"):
                copy_fixed_validation_samples(fixed_dir, output_dir)


if __name__ == "__main__":
    unittest.main()
