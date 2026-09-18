import pathlib
import sys
import unittest
import numpy as np
from PIL import Image

# Ensure repo root is in sys.path
REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import tensorflow as tf
from models import build_gournet, build_gournet_v2
from models.gradcam import find_last_conv_layer, generate_gradcam_heatmap, overlay_gradcam


class TestGradCAM(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m_base = build_gournet(8, (224, 224, 3))
        cls.m_v2 = build_gournet_v2(8, (224, 224, 3))

    def test_find_last_conv_layer(self):
        self.assertEqual(find_last_conv_layer(self.m_base), "Convolution-4")
        self.assertEqual(find_last_conv_layer(self.m_v2), "Block4_Conv")

    def test_find_last_conv_layer_fallback(self):
        # Create a simple dummy model with a custom conv layer
        inputs = tf.keras.Input(shape=(32, 32, 3))
        x = tf.keras.layers.Conv2D(16, 3, name="custom_conv_1")(inputs)
        x = tf.keras.layers.Dense(4, name="custom_dense")(x)
        dummy_model = tf.keras.Model(inputs, x)
        self.assertEqual(find_last_conv_layer(dummy_model), "custom_conv_1")

    def test_find_last_conv_layer_not_found(self):
        # Create a model without conv layers
        inputs = tf.keras.Input(shape=(10,))
        x = tf.keras.layers.Dense(4)(inputs)
        dummy_model = tf.keras.Model(inputs, x)
        with self.assertRaises(ValueError):
            find_last_conv_layer(dummy_model)

    def test_generate_gradcam_heatmap(self):
        batch = np.zeros((1, 224, 224, 3), dtype=np.float32)
        heatmap = generate_gradcam_heatmap(self.m_base, batch, pred_index=0)
        self.assertEqual(heatmap.shape, (224, 224))
        self.assertTrue(0.0 <= np.min(heatmap) <= np.max(heatmap) <= 1.0)

    def test_generate_gradcam_heatmap_v2_auto_pred_index(self):
        batch = np.random.uniform(0.0, 1.0, size=(1, 224, 224, 3)).astype(np.float32)
        heatmap = generate_gradcam_heatmap(self.m_v2, batch, pred_index=None)
        self.assertEqual(heatmap.shape, (224, 224))
        self.assertTrue(0.0 <= np.min(heatmap) <= np.max(heatmap) <= 1.0)

    def test_generate_gradcam_heatmap_constant_activation(self):
        # Edge case: zero activation or constant
        batch = np.zeros((1, 224, 224, 3), dtype=np.float32)
        heatmap = generate_gradcam_heatmap(self.m_v2, batch, pred_index=0)
        self.assertEqual(heatmap.shape, (224, 224))
        self.assertFalse(np.isnan(heatmap).any())
        self.assertTrue(0.0 <= np.min(heatmap) <= np.max(heatmap) <= 1.0)

    def test_overlay_gradcam(self):
        img = Image.new("RGB", (224, 224), color=(34, 139, 34))
        heatmap = np.ones((224, 224), dtype=np.float32) * 0.5
        blended = overlay_gradcam(img, heatmap, alpha=0.5, colormap_name="jet")
        self.assertEqual(blended.size, (224, 224))
        self.assertEqual(blended.mode, "RGB")

    def test_overlay_gradcam_resize_non_standard_image(self):
        img = Image.new("RGB", (100, 150), color=(10, 20, 30))
        heatmap = np.zeros((224, 224), dtype=np.float32)
        blended = overlay_gradcam(img, heatmap, alpha=0.4, colormap_name="viridis")
        self.assertEqual(blended.size, (224, 224))
        self.assertEqual(blended.mode, "RGB")

    def test_gradcam_determinism_training_false(self):
        # Non-zero structured image to verify no random augmentation occurs
        rng = np.random.default_rng(42)
        batch = rng.uniform(0.0, 255.0, (1, 224, 224, 3)).astype(np.float32)
        h1 = generate_gradcam_heatmap(self.m_base, batch, pred_index=0)
        h2 = generate_gradcam_heatmap(self.m_base, batch, pred_index=0)
        np.testing.assert_allclose(h1, h2, atol=1e-6, err_msg="Grad-CAM must be deterministic with training=False")

        h_v2_1 = generate_gradcam_heatmap(self.m_v2, batch, pred_index=1)
        h_v2_2 = generate_gradcam_heatmap(self.m_v2, batch, pred_index=1)
        np.testing.assert_allclose(h_v2_1, h_v2_2, atol=1e-6, err_msg="GourNet v2 Grad-CAM must be deterministic")


if __name__ == "__main__":
    unittest.main()
