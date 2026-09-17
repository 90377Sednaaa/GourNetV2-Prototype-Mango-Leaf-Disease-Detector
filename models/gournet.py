"""
GourNet (baseline) — faithful recreation of Alam et al. (2026),
"GourNet: A CNN-Based Model for Mango Leaf Disease Detection" (AdComSys 2025).

Architecture matches Figure 4 in the paper:
4x [Conv2D -> ReLU -> MaxPool] -> Flatten -> Dense(64) -> Dense(num_classes)
Total parameters at 224x224x3 input, 8 classes: 683,656 (verified against the paper).
"""

import tensorflow as tf
from tensorflow.keras import layers, models

IMG_SIZE = (224, 224)
NUM_CLASSES = 8


def build_gournet(num_classes: int = NUM_CLASSES, input_shape=(224, 224, 3)) -> tf.keras.Model:
    # Sequential-1: rescaling (0-255 -> 0-1)
    rescale = tf.keras.Sequential([layers.Rescaling(1.0 / 255)], name="Sequential-1")

    # Sequential-2: augmentation (random flip + random rotation, per paper text)
    # NOTE: augmentation layers are no-ops at inference time (model.predict / training=False),
    # so they are safe to keep in the graph for a loaded/inference-only model.
    augment = tf.keras.Sequential(
        [layers.RandomFlip("horizontal_and_vertical"), layers.RandomRotation(0.2)],
        name="Sequential-2",
    )

    inputs = tf.keras.Input(shape=input_shape)
    x = rescale(inputs)
    x = augment(x)

    x = layers.Conv2D(32, 3, activation="relu", name="Convolution-1")(x)
    x = layers.MaxPooling2D(pool_size=2, name="Max_Pooling-1")(x)

    x = layers.Conv2D(64, 3, activation="relu", name="Convolution-2")(x)
    x = layers.MaxPooling2D(pool_size=2, name="Max_Pooling-2")(x)

    x = layers.Conv2D(64, 3, activation="relu", name="Convolution-3")(x)
    x = layers.MaxPooling2D(pool_size=2, name="Max_Pooling-3")(x)

    x = layers.Conv2D(64, 3, activation="relu", name="Convolution-4")(x)
    x = layers.MaxPooling2D(pool_size=2, name="Max_Pooling-4")(x)

    x = layers.Flatten(name="Flatten")(x)
    x = layers.Dense(64, activation="relu", name="Dense-1")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="Dense-2")(x)

    return models.Model(inputs, outputs, name="GourNet")
