import tensorflow as tf
from typing import List, Any
from .sampling import downsampler
import matplotlib.pyplot as plt
import numpy as np


class PatchGANDiscriminator(tf.keras.models.Model):

    def __init__(self, input_dim: int = 300, seed: Any = None, comparing: bool = True):
        super(PatchGANDiscriminator, self).__init__()

        self.comparing = comparing

        if comparing:
            self.concatenate_layer = tf.keras.layers.Concatenate()  # (batch_size, 300, 300, 6)

        self.downsamplers: List[tf.keras.layers.Layer] = [
            downsampler(64, 4, apply_bn=False, seed=seed, input_shape=(input_dim, input_dim, 6)),  # (batch_size, 150, 150, 64)
            downsampler(128, 4, seed=seed),  # (batch_size, 75, 75, 128)
            downsampler(256, 4, seed=seed)  # (batch_size, 38, 38, 256)
        ]

        initializer = tf.random_normal_initializer(0., 0.02)
        self.conv_layer = tf.keras.layers.Conv2D(512, 6, strides=1, kernel_initializer=initializer,
                                                 use_bias=False)   # (batch_size, 33, 33, 512)

        self.bn = tf.keras.layers.BatchNormalization()

        self.relu = tf.keras.layers.LeakyReLU()

        self.zero_pad2 = tf.keras.layers.ZeroPadding2D()  # (batch_size, 35, 35, 256)

        self.last_layer = tf.keras.layers.Conv2D(1, 4, strides=1,
                                                 kernel_initializer=initializer)  # (batch_size, 32, 32, 1)

    def call(self, inputs: List, training=None, mask=None):

        x = inputs

        if self.comparing:
            x = self.concatenate_layer(x)

        for down in self.downsamplers:
            x = down(x)

        x = self.conv_layer(x)
        x = self.bn(x)
        x = self.zero_pad2(x)

        return self.last_layer(x)


if __name__ == "__main__":

    DIM = 300
    SHAPE = (DIM, DIM, 3)
    disc = PatchGANDiscriminator(input_dim=DIM, seed=666)
    sample = np.array(plt.imread("../processed/300x300/satellite/0063.jpg"), dtype="uint8")
    img = disc.predict([sample[tf.newaxis], sample[tf.newaxis]], batch_size=1)
    plt.subplot(1, 3, 1)
    plt.imshow(sample)
    plt.title("Original image")
    plt.subplot(1, 3, 2)
    plt.imshow(sample)
    plt.title("Compared image")
    plt.subplot(1, 3, 3)
    plt.imshow(img.reshape((32, 32, 1)))
    plt.title("Untrained discriminator output")
    plt.show()
