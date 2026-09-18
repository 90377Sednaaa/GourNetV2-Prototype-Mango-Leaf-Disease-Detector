import pathlib
import sys
import tempfile
import unittest
import numpy as np

# Ensure repo root is on sys.path
REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ui.components import (
    get_sample_images,
    format_consensus_data,
    render_hero_header,
    render_model_status_ribbon,
    render_consensus_banner,
    render_comparative_ribbon,
    render_model_card,
)


class TestUIComponents(unittest.TestCase):
    def test_get_sample_images_empty_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(get_sample_images(pathlib.Path(tmp)), [])

    def test_get_sample_images_nonexistent_dir(self):
        nonexistent = pathlib.Path("non_existent_dir_12345")
        self.assertEqual(get_sample_images(nonexistent), [])

    def test_get_sample_images_filters_and_sorts(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = pathlib.Path(tmp)
            (p / "b_leaf.jpg").write_bytes(b"123")
            (p / "a_leaf.png").write_bytes(b"456")
            (p / "c_leaf.jpeg").write_bytes(b"789")
            (p / "d_leaf.JPG").write_bytes(b"000")
            (p / "readme.txt").write_bytes(b"txt")
            imgs = get_sample_images(p)
            self.assertEqual(len(imgs), 4)
            self.assertEqual(imgs[0].name, "a_leaf.png")
            self.assertEqual(imgs[1].name, "b_leaf.jpg")
            self.assertEqual(imgs[2].name, "c_leaf.jpeg")
            self.assertEqual(imgs[3].name, "d_leaf.JPG")

    def test_format_consensus_data_agreement(self):
        data = format_consensus_data("Healthy", "Healthy", 0.95, 0.98, "GourNet", "GourNet v2")
        self.assertTrue(data["agreed"])
        self.assertEqual(data["css_class"], "consensus-banner")
        self.assertIn("Healthy", data["title"])
        self.assertIn("95.0%", data["detail"])
        self.assertIn("98.0%", data["detail"])

    def test_format_consensus_data_disagreement(self):
        data = format_consensus_data("Healthy", "Powdery Mildew", 0.70, 0.85, "GourNet", "GourNet v2")
        self.assertFalse(data["agreed"])
        self.assertEqual(data["css_class"], "divergence-banner")
        self.assertIn("Models Disagree", data["title"])
        self.assertIn("Healthy", data["detail"])
        self.assertIn("Powdery Mildew", data["detail"])

    def test_render_functions_execute(self):
        # Verify render functions execute without error in bare/headless mode
        render_hero_header()

        static_infos = {
            "GourNet (baseline)": (683656, 8.3),
            "GourNet v2 (enhanced)": (98280, 1.3),
        }
        trained_flags = {
            "GourNet (baseline)": True,
            "GourNet v2 (enhanced)": True,
        }
        registry = {
            "GourNet (baseline)": {"short": "GourNet", "notes": "4-conv-block CNN."},
            "GourNet v2 (enhanced)": {"short": "GourNet v2", "notes": "GroupNorm + GAP variant."},
        }
        render_model_status_ribbon(static_infos, trained_flags, "Standard", registry)

        consensus_data = format_consensus_data("Healthy", "Healthy", 0.95, 0.98, "GourNet", "GourNet v2")
        render_consensus_banner(consensus_data)
        render_consensus_banner("Healthy", "Anthracnose", 0.8, 0.7, "GourNet", "GourNet v2")

        render_comparative_ribbon(
            a_ms=15.2,
            b_ms=12.4,
            a_conf=0.95,
            b_conf=0.98,
            a_margin=0.90,
            b_margin=0.95,
            short_a="GourNet",
            short_b="GourNet v2",
        )

        class_names = ["Anthracnose", "Healthy", "Powdery Mildew"]
        probs = np.array([0.05, 0.90, 0.05])
        res = {
            "probs": probs,
            "ms": 14.5,
            "stats": {
                "top_idx": 1,
                "conf": 0.90,
                "margin": 0.85,
                "entropy": 0.35,
                "order": np.array([1, 0, 2]),
            },
        }
        render_model_card("GourNet v2", "GourNet v2 (enhanced)", res, 3, class_names)
        # Test edge case: empty or None res
        render_model_card("GourNet v2", "GourNet v2 (enhanced)", {}, 3, class_names)
        render_model_card("GourNet v2", "GourNet v2 (enhanced)", None, 3, class_names)


if __name__ == "__main__":
    unittest.main()
