"""
Grad-CAM (Gradient-weighted Class Activation Mapping) Computation Engine.

Provides visual explanations for CNN decisions by computing gradient-weighted
activation maps of the final convolutional layer with respect to predicted classes.
"""

from __future__ import annotations

import matplotlib
import numpy as np
from PIL import Image
import tensorflow as tf


def find_last_conv_layer(model: tf.keras.Model) -> str:
    """
    Identifies the name of the final convolutional layer in the given model.

    Checks specifically for known GourNet architecture layer names ('Convolution-4',
    'Block4_Conv'), falling back to scanning backwards through the model's layers
    for any tf.keras.layers.Conv2D layer.

    Args:
        model: A compiled or uncompiled tf.keras.Model.

    Returns:
        The string name of the target convolutional layer.

    Raises:
        ValueError: If no convolutional layer is present in the model.
    """
    layer_names = {layer.name for layer in model.layers}
    if "Convolution-4" in layer_names:
        return "Convolution-4"
    if "Block4_Conv" in layer_names:
        return "Block4_Conv"

    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name

    raise ValueError(f"No convolutional layer found in model '{model.name}'.")


def generate_gradcam_heatmap(
    model: tf.keras.Model,
    batch: np.ndarray,
    last_conv_layer_name: str | None = None,
    pred_index: int | None = None,
) -> np.ndarray:
    """
    Generates a 2D Grad-CAM activation heatmap for an input image.

    Args:
        model: A trained tf.keras.Model.
        batch: Input image tensor or numpy array, shape (1, 224, 224, 3) or (224, 224, 3).
        last_conv_layer_name: Optional name of the convolutional layer to probe.
                              If None, auto-detected via find_last_conv_layer.
        pred_index: Target class index to explain. If None, the top predicted class is used.

    Returns:
        A 2D float32 numpy array with shape (224, 224) normalized to [0.0, 1.0].
    """
    if last_conv_layer_name is None:
        last_conv_layer_name = find_last_conv_layer(model)

    if isinstance(batch, np.ndarray):
        batch_tensor = tf.convert_to_tensor(batch, dtype=tf.float32)
    else:
        batch_tensor = tf.cast(batch, tf.float32)

    if tf.rank(batch_tensor) == 3:
        batch_tensor = tf.expand_dims(batch_tensor, axis=0)

    conv_layer = model.get_layer(last_conv_layer_name)
    grad_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=[conv_layer.output, model.output],
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(batch_tensor, training=False)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weighted sum of conv feature maps: conv_outputs[0] has shape (H, W, C), pooled_grads has (C,)
    conv_outputs_sample = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_outputs_sample * pooled_grads, axis=-1)

    # ReLU to focus on features having positive influence on the target class
    heatmap = tf.nn.relu(heatmap)

    # Resize to (224, 224) via bilinear interpolation
    heatmap_resized = tf.image.resize(
        tf.expand_dims(heatmap, axis=-1),
        (224, 224),
        method=tf.image.ResizeMethod.BILINEAR,
    )
    heatmap_2d = tf.squeeze(heatmap_resized).numpy()

    # Normalize to [0.0, 1.0] by dividing by (max - min + 1e-8)
    h_min = float(np.min(heatmap_2d))
    h_max = float(np.max(heatmap_2d))
    heatmap_norm = (heatmap_2d - h_min) / (h_max - h_min + 1e-8)
    return np.clip(heatmap_norm, 0.0, 1.0).astype(np.float32)


def overlay_gradcam(
    img: Image.Image,
    heatmap: np.ndarray,
    alpha: float = 0.5,
    colormap_name: str = "jet",
) -> Image.Image:
    """
    Overlays a Grad-CAM heatmap onto a PIL image using a matplotlib colormap.

    Args:
        img: Input PIL Image.
        heatmap: 2D numpy array of shape (224, 224) with values in [0.0, 1.0].
        alpha: Blending weight for the heatmap overlay (0.0 = original image only, 1.0 = heatmap only).
        colormap_name: Name of matplotlib colormap (default 'jet').

    Returns:
        Blended PIL Image in RGB format of size (224, 224).
    """
    if img.size != (224, 224):
        img = img.resize((224, 224), resample=Image.Resampling.BILINEAR)

    img_rgb = img.convert("RGB")
    img_arr = np.array(img_rgb, dtype=np.float32)

    cmap = matplotlib.colormaps[colormap_name]
    cam_rgba = cmap(heatmap)
    cam_rgb = cam_rgba[..., :3] * 255.0

    blended = (1.0 - alpha) * img_arr + alpha * cam_rgb
    blended_uint8 = np.clip(blended, 0, 255).astype(np.uint8)

    return Image.fromarray(blended_uint8)
