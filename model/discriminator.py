import tensorflow as tf
from typing import List, Any
from .model_components import downsampler
import tensorflow_addons as tfda


class PatchGANDiscriminator(tf.keras.models.Model):

    def __init__(self, input_dim: int = 300, seed: Any = None, comparing: bool = True, add_noise: bool = False,
                 instance_normalization: bool = False):
        super(PatchGANDiscriminator, self).__init__()

        channels = 3

        self.noisy = add_noise

        if self.noisy:
            self.noise_layer = tf.keras.layers.GaussianNoise(stddev=1, input_shape=(input_dim, input_dim, channels))

        self.comparing = comparing

        if comparing:
            self.concatenate_layer = tf.keras.layers.Concatenate()  # (batch_size, 300, 300, 6)
            channels = 6

        self.downsamplers: List[tf.keras.layers.Layer] = [
            downsampler(64, 4, apply_bn=False, seed=seed, input_shape=(input_dim, input_dim, channels)),  # (batch_size, 150, 150, 64)
            downsampler(128, 4, seed=seed),  # (batch_size, 75, 75, 128)
            downsampler(256, 4, seed=seed)  # (batch_size, 38, 38, 256)
        ]

        initializer = tf.random_normal_initializer(0., 0.02)
        self.conv_layer = tf.keras.layers.Conv2D(512, 4, strides=1, kernel_initializer=initializer,
                                                 use_bias=False)   # (batch_size, 33, 33, 512)

        if instance_normalization:
            self.bn = tfda.layers.InstanceNormalization()
        else:
            self.bn = tf.keras.layers.BatchNormalization()

        self.relu = tf.keras.layers.LeakyReLU()

        self.zero_pad2 = tf.keras.layers.ZeroPadding2D()  # (batch_size, 35, 35, 256)

        self.last_layer = tf.keras.layers.Conv2D(1, 4, strides=1,
                                                 kernel_initializer=initializer)  # (batch_size, 32, 32, 1)

    def call(self, inputs, training=None, mask=None):

        x = inputs

        if self.noisy:
            x = self.noise_layer(x)

        if self.comparing:
            x = self.concatenate_layer(x)

        for down in self.downsamplers:
            x = down(x)

        x = self.conv_layer(x)
        x = self.bn(x)
        x = self.zero_pad2(x)

        return self.last_layer(x)


if __name__ == "__main__":

    pass
