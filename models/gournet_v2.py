"""
GourNet v2 — enhanced variant of the baseline GourNet.

Changes vs. the paper's original architecture:
- BatchNormalization -> GroupNormalization after every conv block and the dense head
  (fixes an unstable-batch-statistics issue seen with small step counts / early stopping).
- Flatten -> GlobalAveragePooling2D, plus Dropout, before the dense head (fewer params,
  less overfitting).
Total parameters at 224x224x3 input, 8 classes: 98,280 (~7x fewer than the baseline).
"""

import tensorflow as tf
from tensorflow.keras import layers, models

IMG_SIZE = (224, 224)
NUM_CLASSES = 8


def build_gournet_v2(num_classes: int = NUM_CLASSES, input_shape=(224, 224, 3)) -> tf.keras.Model:
    rescale = tf.keras.Sequential([layers.Rescaling(1.0 / 255)], name="Sequential-1")

    # NOTE: augmentation layers are no-ops at inference time (model.predict / training=False),
    # so they are safe to keep in the graph for a loaded/inference-only model.
    augment = tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal_and_vertical"),
            layers.RandomRotation(0.2),
        ],
        name="Sequential-2",
    )

    inputs = tf.keras.Input(shape=input_shape)
    x = rescale(inputs)
    x = augment(x)

    def conv_bn_block(x, filters, name_prefix):
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False, name=f"{name_prefix}_Conv")(x)
        x = layers.GroupNormalization(groups=8, name=f"{name_prefix}_GN")(x)
        x = layers.Activation("relu", name=f"{name_prefix}_ReLU")(x)
        x = layers.MaxPooling2D(pool_size=2, name=f"{name_prefix}_Pool")(x)
        return x

    x = conv_bn_block(x, 32, "Block1")
    x = conv_bn_block(x, 64, "Block2")
    x = conv_bn_block(x, 64, "Block3")
    x = conv_bn_block(x, 64, "Block4")

    x = layers.Dropout(0.2, name="Spatial_Dropout")(x)
    x = layers.GlobalAveragePooling2D(name="GAP")(x)

    x = layers.Dense(64, name="Dense-1")(x)
    x = layers.GroupNormalization(groups=8, name="Dense-1_GN")(x)
    x = layers.Activation("relu", name="Dense-1_ReLU")(x)
    x = layers.Dropout(0.4, name="Head_Dropout")(x)

    outputs = layers.Dense(num_classes, activation="softmax", name="Dense-2")(x)

    return models.Model(inputs, outputs, name="GourNet_v2")
