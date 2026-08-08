from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from preprocess import (
    build_tier_parent_context,
    compose_tier_context_patch,
    get_safe_size,
    load_image,
    load_map_export_manifest,
    safe_imwrite,
)


class MapExportContractTests(unittest.TestCase):
    def test_manifest_loads_against_source_images(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source_images"
            source.mkdir()
            base = np.zeros((16, 16, 4), dtype=np.uint8)
            base[..., 3] = 255
            tier = np.zeros((8, 8, 4), dtype=np.uint8)
            tier[2:6, 2:6] = (255, 255, 255, 255)
            safe_imwrite(source / "Map02Base.png", base)
            safe_imwrite(source / "Map02Lv002Tier255.png", tier)

            manifest = {
                "format": "map-cls-export-v1",
                "bases": {
                    "Map02Base.png": {
                        "file": "Map02Base.png",
                        "size": [16, 16],
                    }
                },
                "tiers": {
                    "Map02Lv002Tier255": {
                        "template": "Map02Lv002Tier255.png",
                        "parent": "Map02Base.png",
                        "template_size": [8, 8],
                        "parent_size": [16, 16],
                        "tier_to_parent": [1.0, 0.0, 1.0, 0.0],
                        "mask_mode": "opaque",
                    }
                },
            }
            manifest_path = root / "map_export.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            specs = load_map_export_manifest(manifest_path, source)

            self.assertEqual(specs["Map02Lv002Tier255"]["parent_size"], (16, 16))
            self.assertEqual(specs["Map02Lv002Tier255"]["template_size"], (8, 8))

    def test_tier_context_preserves_parent_pixels_outside_foreground(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            parent = np.zeros((64, 64, 4), dtype=np.uint8)
            parent[..., 0] = 80
            parent[..., 1] = 120
            parent[..., 3] = 255
            parent_path = root / "Map02Base.png"
            safe_imwrite(parent_path, parent)

            template_path = root / "Map02Lv002Tier255.png"
            template = np.zeros((32, 32, 4), dtype=np.uint8)
            template[..., 3] = 255
            template[12:20, 12:20] = (255, 255, 255, 255)
            safe_imwrite(template_path, template)
            spec = {
                "parent_path": str(parent_path),
                "parent_size": (64, 64),
                "affine": (1.0, 16.0, 1.0, 16.0),
                "mask_mode": "opaque",
            }

            padded_template = load_image(template_path, get_safe_size())
            context = build_tier_parent_context(padded_template, spec, get_safe_size())
            output = compose_tier_context_patch(
                padded_template,
                context["parent_aligned"],
            )
            center = get_safe_size() // 2

            self.assertGreater(int(output[center + 8, center + 8].sum()), 0)
            self.assertGreater(int(output[center + 16, center + 16].sum()), 0)
            np.testing.assert_array_equal(
                output[center + 20, center + 20],
                context["parent_aligned"][center + 20, center + 20],
            )


if __name__ == "__main__":
    unittest.main()
