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
from models.gradcam import generate_gradcam_heatmap, overlay_gradcam


class TestEndToEnd(unittest.TestCase):
    def test_end_to_end_inference_standard_weights(self):
        # Create a synthetic mango leaf image
        img = Image.new("RGB", (224, 224), color=(34, 139, 34))
        batch = app.preprocess_image(img)
        self.assertEqual(batch.shape, (1, 224, 224, 3))

        # Load models for Standard variant
        m_base, ok_base = app.load_model(app.MODEL_KEYS[0], "Standard")
        m_v2, ok_v2 = app.load_model(app.MODEL_KEYS[1], "Standard")
        self.assertTrue(ok_base, "Baseline model Standard weights should load successfully")
        self.assertTrue(ok_v2, "GourNet v2 Standard weights should load successfully")

        # Predict
        p_base, dt_base = app.timed_predict(m_base, batch)
        p_v2, dt_v2 = app.timed_predict(m_v2, batch)

        self.assertEqual(len(p_base), 8)
        self.assertEqual(len(p_v2), 8)
        self.assertAlmostEqual(float(np.sum(p_base)), 1.0, places=4)
        self.assertAlmostEqual(float(np.sum(p_v2)), 1.0, places=4)
        self.assertGreater(dt_base, 0)
        self.assertGreater(dt_v2, 0)

        # Verify prediction stats
        s_base = app.prediction_stats(p_base)
        s_v2 = app.prediction_stats(p_v2)
        self.assertIn("top_idx", s_base)
        self.assertIn("margin", s_base)
        self.assertIn("entropy", s_base)
        self.assertIn("top_idx", s_v2)
        self.assertIn("margin", s_v2)
        self.assertIn("entropy", s_v2)

    def test_end_to_end_inference_12k_weights(self):
        img = Image.new("RGB", (224, 224), color=(60, 179, 113))
        batch = app.preprocess_image(img)
        m_base, ok_base = app.load_model(app.MODEL_KEYS[0], "12k")
        m_v2, ok_v2 = app.load_model(app.MODEL_KEYS[1], "12k")
        self.assertTrue(ok_base, "Baseline model 12k weights should load successfully")
        self.assertTrue(ok_v2, "GourNet v2 12k weights should load successfully")
        p_base, dt_base = app.timed_predict(m_base, batch)
        p_v2, dt_v2 = app.timed_predict(m_v2, batch)
        self.assertEqual(len(p_base), 8)
        self.assertEqual(len(p_v2), 8)
        self.assertAlmostEqual(float(np.sum(p_base)), 1.0, places=4)
        self.assertAlmostEqual(float(np.sum(p_v2)), 1.0, places=4)
        self.assertGreater(dt_base, 0)
        self.assertGreater(dt_v2, 0)

    def test_model_static_info(self):
        m_base, _ = app.load_model(app.MODEL_KEYS[0], "Standard")
        m_v2, _ = app.load_model(app.MODEL_KEYS[1], "Standard")
        params_base, size_base = app.model_static_info(
            m_base, app.MODEL_REGISTRY[app.MODEL_KEYS[0]]["weights"]["Standard"]
        )
        params_v2, size_v2 = app.model_static_info(
            m_v2, app.MODEL_REGISTRY[app.MODEL_KEYS[1]]["weights"]["Standard"]
        )
        self.assertGreater(params_base, 0)
        self.assertGreater(params_v2, 0)
        self.assertLess(params_v2, params_base, "GourNet v2 should have fewer parameters than baseline")
        self.assertGreater(size_base, 0)
        self.assertGreater(size_v2, 0)

    def test_end_to_end_gradcam_real_weights(self):
        # Create a synthetic mango leaf image
        img = Image.new("RGB", (224, 224), color=(34, 139, 34))
        batch = app.preprocess_image(img)

        for variant in ["Standard", "12k"]:
            for key in app.MODEL_KEYS:
                model, ok = app.load_model(key, variant)
                self.assertTrue(ok, f"Model {key} ({variant}) should load successfully")

                # Verify Grad-CAM with auto prediction index
                heatmap_auto = generate_gradcam_heatmap(model, batch)
                self.assertEqual(heatmap_auto.shape, (224, 224))
                self.assertEqual(heatmap_auto.dtype, np.float32)
                self.assertGreaterEqual(float(np.min(heatmap_auto)), 0.0)
                self.assertLessEqual(float(np.max(heatmap_auto)), 1.0)

                # Verify Grad-CAM with explicit target class index
                heatmap_explicit = generate_gradcam_heatmap(model, batch, pred_index=0)
                self.assertEqual(heatmap_explicit.shape, (224, 224))
                self.assertEqual(heatmap_explicit.dtype, np.float32)
                self.assertGreaterEqual(float(np.min(heatmap_explicit)), 0.0)
                self.assertLessEqual(float(np.max(heatmap_explicit)), 1.0)

                # Verify overlay generation
                overlay = overlay_gradcam(img, heatmap_auto, alpha=0.5, colormap_name="jet")
                self.assertIsInstance(overlay, Image.Image)
                self.assertEqual(overlay.size, img.size)


if __name__ == "__main__":
    unittest.main()

