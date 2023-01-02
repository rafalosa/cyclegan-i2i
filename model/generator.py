import tensorflow as tf
from typing import List, Any
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

        self.downsamplers: List[tf.keras.layers.Layer] = [
            downsampler(64, 4, apply_bn=False, seed=seed,
                        input_shape=(input_dim, input_dim, 3), name="down1"),
            downsampler(128, 4, seed=seed, name="down2"),
            downsampler(256, 4, seed=seed, name="down3"),
            downsampler(512, 4, seed=seed, name="down4"),
            downsampler(512, 4, seed=seed, name="down5"),
            downsampler(512, 4, seed=seed, name="down6"),
            downsampler(512, 4, seed=seed, name="down7"),
            downsampler(512, 4, seed=seed, name="down8"),
        ]
        self.upsamplers: List[tf.keras.layers.Layer] = [
            transposed_upsampler(512, 4, apply_dropout=True, seed=seed, name="up1"),
            transposed_upsampler(512, 4, apply_dropout=True, seed=seed, name="up2"),
            transposed_upsampler(512, 4, apply_dropout=True, seed=seed, name="up3"),
            transposed_upsampler(512, 4, seed=seed, name="up4"),
            transposed_upsampler(256, 4, seed=seed, name="up5"),
            transposed_upsampler(128, 4, seed=seed, name="up6"),
            transposed_upsampler(64, 4, seed=seed, name="up7"),
        ]

        initializer = tf.random_normal_initializer(0., 0.02, seed=seed)
        self.last_layer = tf.keras.layers.Conv2DTranspose(3, 4,
                                                          strides=2,
                                                          padding='same',
                                                          kernel_initializer=initializer,
                                                          activation='tanh')
