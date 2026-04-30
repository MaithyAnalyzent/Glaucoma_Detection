from tensorflow.keras import Model
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Concatenate, Conv2D, Conv2DTranspose, Input
from tensorflow.keras.optimizers import Adam


def conv_block(x, filters):
    x = Conv2D(filters, 3, activation="relu", padding="same")(x)
    x = Conv2D(filters, 3, activation="relu", padding="same")(x)
    return x


def TL_unet_model(input_shape=(256, 256, 3)):
    """V-UNET V1.1: VGG16 transfer-learning encoder with U-Net decoder."""
    inputs = Input(shape=input_shape)
    base_model = VGG16(include_top=False, weights="imagenet", input_tensor=inputs)
    base_model.trainable = False

    skip1 = base_model.get_layer("block1_conv2").output
    skip2 = base_model.get_layer("block2_conv2").output
    skip3 = base_model.get_layer("block3_conv3").output
    skip4 = base_model.get_layer("block4_conv3").output
    bridge = base_model.get_layer("block5_conv3").output

    d1 = Conv2DTranspose(128, 2, strides=2, padding="same")(bridge)
    d1 = Concatenate()([d1, skip4])
    d1 = conv_block(d1, 128)

    d2 = Conv2DTranspose(64, 2, strides=2, padding="same")(d1)
    d2 = Concatenate()([d2, skip3])
    d2 = conv_block(d2, 64)

    d3 = Conv2DTranspose(32, 2, strides=2, padding="same")(d2)
    d3 = Concatenate()([d3, skip2])
    d3 = conv_block(d3, 32)

    d4 = Conv2DTranspose(16, 2, strides=2, padding="same")(d3)
    d4 = Concatenate()([d4, skip1])
    d4 = conv_block(d4, 16)

    outputs = Conv2D(1, 1, activation="sigmoid", name="segmentation_mask")(d4)
    model = Model(inputs=inputs, outputs=outputs, name="V-UNET_V1_1")
    return model


def compile_unet(model, learning_rate=1e-5, metrics=None):
    if metrics is None:
        metrics = []
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=metrics,
    )
    return model
