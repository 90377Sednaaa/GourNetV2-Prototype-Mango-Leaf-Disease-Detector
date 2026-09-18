import pathlib
import sys
import unittest
import numpy as np
from PIL import Image

# Ensure repo root is in sys.path
REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import app


class TestAppStructure(unittest.TestCase):
    def test_registry_and_constants(self):
        self.assertIn("GourNet (baseline)", app.MODEL_REGISTRY)
        self.assertIn("GourNet v2 (enhanced)", app.MODEL_REGISTRY)
        self.assertEqual(len(app.CLASS_NAMES), 8)
        self.assertEqual(app.IMG_SIZE, (224, 224))
        self.assertIn("Standard", app.WEIGHT_OPTIONS)
        self.assertIn("12k", app.WEIGHT_OPTIONS)

    def test_prediction_stats(self):
        dummy_probs = np.array([0.05, 0.75, 0.05, 0.05, 0.02, 0.03, 0.03, 0.02])
        stats = app.prediction_stats(dummy_probs)
        self.assertEqual(stats["top_idx"], 1)
        self.assertAlmostEqual(stats["conf"], 0.75)
        self.assertGreater(stats["margin"], 0.6)
        self.assertGreater(stats["entropy"], 0)

    def test_preprocess_image(self):
        img = Image.new("RGB", (300, 300), color=(128, 128, 128))
        batch = app.preprocess_image(img)
        self.assertEqual(batch.shape, (1, 224, 224, 3))
        self.assertEqual(batch.dtype, np.float32)


if __name__ == "__main__":
    unittest.main()
