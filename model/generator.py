import tensorflow as tf
from typing import List, Any, Literal
from .model_components import transposed_upsampler, downsampler, unpooling_upsampler, ResidualBlock, ReflectionPadding2D


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

    def __init__(self, input_dim: int = 256, seed: Any = None, upsampling: Literal["upconv", "deconv"] = "deconv",
                 instance_normalization: bool = False):
        super(UNetGenerator, self).__init__()

        self.instance_norm = instance_normalization

        self.downsamplers: List[tf.keras.layers.Layer] = [
            downsampler(64, 4, apply_bn=False, seed=seed,
                        input_shape=(input_dim, input_dim, 3), name="down1",
                        instance_normalization=self.instance_norm),
            downsampler(128, 4, seed=seed, name="down2", instance_normalization=self.instance_norm),
            downsampler(256, 4, seed=seed, name="down3", instance_normalization=self.instance_norm),
            downsampler(512, 4, seed=seed, name="down4", instance_normalization=self.instance_norm),
            downsampler(512, 4, seed=seed, name="down5", instance_normalization=self.instance_norm),
            downsampler(512, 4, seed=seed, name="down6", instance_normalization=self.instance_norm),
            downsampler(512, 4, seed=seed, name="down7", instance_normalization=self.instance_norm),
            downsampler(512, 4, seed=seed, name="down8", instance_normalization=self.instance_norm),
        ]

        if upsampling == "deconv":
            self.upsamplers: List[tf.keras.layers.Layer] = [
                transposed_upsampler(512, 4, apply_dropout=True, seed=seed, name="up1", instance_normalization=self.instance_norm),
                transposed_upsampler(512, 4, apply_dropout=True, seed=seed, name="up2", instance_normalization=self.instance_norm),
                transposed_upsampler(512, 4, apply_dropout=True, seed=seed, name="up3", instance_normalization=self.instance_norm),
                transposed_upsampler(512, 4, seed=seed, name="up4", instance_normalization=self.instance_norm),
                transposed_upsampler(256, 4, seed=seed, name="up5", instance_normalization=self.instance_norm),
                transposed_upsampler(128, 4, seed=seed, name="up6", instance_normalization=self.instance_norm),
                transposed_upsampler(64, 4, seed=seed, name="up7", instance_normalization=self.instance_norm),
            ]

        else:
            self.upsamplers: List[tf.keras.layers.Layer] = [
                unpooling_upsampler(512, 4, apply_dropout=True, seed=seed, name="up1", instance_normalization=self.instance_norm),
                unpooling_upsampler(512, 4, apply_dropout=True, seed=seed, name="up2", instance_normalization=self.instance_norm),
                unpooling_upsampler(512, 4, apply_dropout=True, seed=seed, name="up3", instance_normalization=self.instance_norm),
                unpooling_upsampler(512, 4, seed=seed, name="up4", instance_normalization=self.instance_norm),
                unpooling_upsampler(256, 4, seed=seed, name="up5", instance_normalization=self.instance_norm),
                unpooling_upsampler(128, 4, seed=seed, name="up6", instance_normalization=self.instance_norm),
                unpooling_upsampler(64, 4, seed=seed, name="up7", instance_normalization=self.instance_norm),
            ]

        initializer = tf.random_normal_initializer(0., 0.02, seed=seed)
        self.last_layer = tf.keras.layers.Conv2DTranspose(3, 2,
                                                          strides=2,
                                                          padding='same',
                                                          kernel_initializer=initializer,
                                                          activation='tanh')


class ResNetGenerator(tf.keras.models.Model):

    def __init__(self, input_dim: int = 256, residual_blocks: int = 9, filters: int = 64,
                 seed: Any = None, padding_type: Literal["default", "reflect"] = "default",
                 instance_normalization: bool = False):
        super(ResNetGenerator, self).__init__()

        initializer = tf.random_normal_initializer(0., 0.02, seed=seed)

        self.instance_norm = instance_normalization

        self.pad_1 = ReflectionPadding2D(padding=(3, 3))

        self.conv_1 = tf.keras.layers.Conv2D(filters, kernel_size=(7, 7), kernel_initializer=initializer, activation="relu",
                                             input_shape=(input_dim, input_dim, 3), use_bias=False)

        self.downsamplers: List[tf.keras.layers.Layer] = [
            downsampler(filters * 2, 4, name="down1", seed=seed, apply_bn=True, instance_normalization=self.instance_norm),
            downsampler(filters * 4, 4, name="down2", seed=seed, instance_normalization=self.instance_norm)
        ]

        self.residual_blocks: List[tf.keras.layers.Layer] = [ResidualBlock(filters * 4, 3, seed=seed, name=f"res{i}", padding=padding_type, instance_normalization=self.instance_norm) for i in range(residual_blocks)]

        self.upsamplers: List[tf.keras.layers.Layer] = [
            transposed_upsampler(filters * 2, 4, seed=seed, instance_normalization=self.instance_norm),
            transposed_upsampler(filters, 4, seed=seed, instance_normalization=self.instance_norm)
        ]

        initializer = tf.random_normal_initializer(0., 0.02, seed=seed)

        self.pad_2 = ReflectionPadding2D(padding=(3, 3))

        self.last_layer = tf.keras.layers.Conv2D(3, kernel_size=(7, 7),
                                                 kernel_initializer=initializer,
                                                 activation='tanh')

    def call(self, inputs, training=None, mask=None):

        x = self.pad_1(inputs)

        x = self.conv_1(x, training=training)

        for down in self.downsamplers:
            x = down(x, training=training)

        for residual in self.residual_blocks:
            x = residual(x, training=training)

        for up in self.upsamplers:
            x = up(x, training=training)

        x = self.pad_2(x)

        return self.last_layer(x, training=training)


if __name__ == "__main__":
    pass
