import tensorflow as tf
from typing import List, Any, Optional
import numpy as np
import matplotlib.pyplot as plt
from .sampling import transposed_upsampler, downsampler


class UNetGenerator(tf.keras.models.Model):

    def call(self, inputs, training=None, mask=None):

        skips = []
        x = self.downsamplers[0](inputs, training=training)
        skips.append(x)

        for i, down in enumerate(self.downsamplers[1:]):

            x = down(x, training=training)
            skips.append(x)

        skips = reversed(skips[:-1])

        for up, skip in zip(self.upsamplers, skips):
            x = up(x, training=training)
            x = tf.keras.layers.Concatenate()([x, skip])

        x = self.last_layer(x, training=training)

        return x

    def __init__(self, input_dim: int = 256, seed: Any = None):
        super(UNetGenerator, self).__init__()

        # self.downsamplers: List[tf.keras.layers.Layer] = [
        #     downsampler(64, 5, padding="valid", apply_bn=False, seed=seed,
        #                 input_shape=(input_dim, input_dim, 3), name="down1"),  # (batch_size, 148, 148, 64)
        #     downsampler(128, 3, seed=seed, name="down2"),  # (batch_size, 74, 74, 128)
        #     downsampler(256, 4, padding="valid", seed=seed, name="down3"),  # (batch_size, 36, 36, 256)
        #     downsampler(512, 4, seed=seed, name="down4"),  # (batch_size, 18, 18, 512)
        #     downsampler(512, 3, padding="valid", seed=seed, name="down5"),  # (batch_size, 8, 8, 512)
        #     downsampler(512, 4, seed=seed, name="down6"),  # (batch_size, 4, 4, 512)
        #     downsampler(512, 4, seed=seed, name="down7"),  # (batch_size, 2, 2, 512)
        #     downsampler(512, 4, seed=seed, name="down8"),  # (batch_size, 1, 1, 512)
        # ]
        # self.upsamplers: List[tf.keras.layers.Layer] = [
        #     transposed_upsampler(512, 4, apply_dropout=True, seed=seed, name="up1"),  # (batch_size, 2, 2, 1024)
        #     transposed_upsampler(512, 4, apply_dropout=True, seed=seed, name="up2"),  # (batch_size, 4, 4, 1024)
        #     transposed_upsampler(512, 4, apply_dropout=True, seed=seed, name="up3"),  # (batch_size, 8, 8, 1024)
        #     transposed_upsampler(512, 4, padding="valid", seed=seed, name="up4"),  # (batch_size, 18, 18, 1024)
        #     transposed_upsampler(256, 4, seed=seed, name="up5"),  # (batch_size, 36, 36, 512)
        #     transposed_upsampler(128, 4, padding="valid", seed=seed, name="up6"),  # (batch_size, 74, 74, 256)
        #     transposed_upsampler(64, 4, seed=seed, name="up7"),  # (batch_size, 148, 148, 128)
        # ]

        self.downsamplers: List[tf.keras.layers.Layer] = [
            downsampler(64, 4, apply_bn=False, seed=seed,
                        input_shape=(input_dim, input_dim, 3), name="down1"),  # (batch_size, 148, 148, 64)
            downsampler(128, 4, seed=seed, name="down2"),  # (batch_size, 74, 74, 128)
            downsampler(256, 4, seed=seed, name="down3"),  # (batch_size, 36, 36, 256)
            downsampler(512, 4, seed=seed, name="down4"),  # (batch_size, 18, 18, 512)
            downsampler(512, 4, seed=seed, name="down5"),  # (batch_size, 8, 8, 512)
            downsampler(512, 4, seed=seed, name="down6"),  # (batch_size, 4, 4, 512)
            downsampler(512, 4, seed=seed, name="down7"),  # (batch_size, 2, 2, 512)
            downsampler(512, 4, seed=seed, name="down8"),  # (batch_size, 1, 1, 512)
        ]
        self.upsamplers: List[tf.keras.layers.Layer] = [
            transposed_upsampler(512, 4, apply_dropout=True, seed=seed, name="up1"),  # (batch_size, 2, 2, 1024)
            transposed_upsampler(512, 4, apply_dropout=True, seed=seed, name="up2"),  # (batch_size, 4, 4, 1024)
            transposed_upsampler(512, 4, apply_dropout=True, seed=seed, name="up3"),  # (batch_size, 8, 8, 1024)
            transposed_upsampler(512, 4, seed=seed, name="up4"),  # (batch_size, 18, 18, 1024)
            transposed_upsampler(256, 4, seed=seed, name="up5"),  # (batch_size, 36, 36, 512)
            transposed_upsampler(128, 4, seed=seed, name="up6"),  # (batch_size, 74, 74, 256)
            transposed_upsampler(64, 4, seed=seed, name="up7"),  # (batch_size, 148, 148, 128)
        ]

        initializer = tf.random_normal_initializer(0., 0.02, seed=seed)
        self.last_layer = tf.keras.layers.Conv2DTranspose(3, 6,
                                                          strides=2,
                                                          padding='valid',
                                                          kernel_initializer=initializer,
                                                          activation='tanh')  # (batch_size, 300, 300, 3)


if __name__ == "__main__":

    DIM = 300
    SHAPE = (DIM, DIM, 3)

    gen = UNetGenerator(input_dim=DIM, seed=666)

    sample = plt.imread("../processed/300x300/satellite/0063.jpg")

    pred = gen.predict(sample[tf.newaxis], batch_size=1)
    plt.subplot(1, 2, 1)
    plt.imshow(sample)
    plt.title("Original image")
    plt.subplot(1, 2, 2)
    plt.imshow(pred.reshape(SHAPE))
    plt.title("Untrained generator output")
    plt.show()
