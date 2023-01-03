import tensorflow as tf
from typing import Optional, Any, Tuple
import tensorflow_addons as tfda


def downsampler(filter_num: int, filter_size: int, stride: int = 2, input_shape: Optional[tuple] = None,
                padding: str = "same", apply_bn: bool = True, seed: Any = None, name=None,
                instance_normalization: bool = False):

    initializer = tf.random_normal_initializer(0., 0.02, seed=seed)
    result = tf.keras.Sequential(name=name)
    if input_shape is not None:
        result.add(
            tf.keras.layers.Conv2D(filter_num, filter_size, input_shape=input_shape, strides=stride, padding=padding,
                                   kernel_initializer=initializer, use_bias=False))
    else:
        result.add(
            tf.keras.layers.Conv2D(filter_num, filter_size, strides=stride, padding=padding,
                                   kernel_initializer=initializer, use_bias=False))

    if apply_bn:
        if instance_normalization:
            result.add(tfda.layers.InstanceNormalization())
        else:
            result.add(tf.keras.layers.BatchNormalization())

    result.add(tf.keras.layers.LeakyReLU())

    return result


def transposed_upsampler(filter_num: int, filter_size: int, stride: int = 2,
                         padding: str = "same", apply_dropout=False, seed: Any = None, name=None,
                         instance_normalization: bool = False):

    initializer = tf.random_normal_initializer(0., 0.02, seed=seed)

    result = tf.keras.Sequential(name=name)
    result.add(tf.keras.layers.Conv2DTranspose(filter_num, filter_size, strides=stride,
                                               padding=padding,
                                               kernel_initializer=initializer,
                                               use_bias=False))

    if instance_normalization:
        result.add(tfda.layers.InstanceNormalization())
    else:
        result.add(tf.keras.layers.BatchNormalization())

    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))

    result.add(tf.keras.layers.ReLU())

    return result


def unpooling_upsampler(filter_num: int, filter_size: int, stride: int = 2,
                        padding: str = "same", apply_dropout=False, seed: Any = None, name=None,
                        instance_normalization: bool = False):

    initializer = tf.random_normal_initializer(0., 0.02, seed=seed)

    result = tf.keras.Sequential(name=name)
    result.add(tf.keras.layers.UpSampling2D(size=(stride*2, stride*2)))
    result.add(tf.keras.layers.Conv2D(filter_num, filter_size, strides=stride,
                                      padding=padding,
                                      kernel_initializer=initializer,
                                      use_bias=False))
    if instance_normalization:
        result.add(tfda.layers.InstanceNormalization())
    else:
        result.add(tf.keras.layers.BatchNormalization())

    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))

    result.add(tf.keras.layers.ReLU())

    return result


class ResidualBlock(tf.keras.layers.Layer):

    def __init__(self, filter_num: int, filter_size: int, seed: Any = None, name=None, padding="default",
                 instance_normalization: bool = False):
        super(ResidualBlock, self).__init__(name=name)
        self.filter_num = filter_num
        self.filter_size = filter_size
        self.seed = seed
        self.padding_type = padding

        self.activation_1 = tf.keras.layers.Activation("linear", trainable=False)
        if padding == "default":
            self.conv_1 = tf.keras.layers.Conv2D(filters=self.filter_num, kernel_size=self.filter_size,
                                                 padding="same", trainable=True)
        elif padding == "reflect":
            self.pad_1 = ReflectionPadding2D(padding=(1, 1))
            self.conv_1 = tf.keras.layers.Conv2D(filters=self.filter_num, kernel_size=self.filter_size,
                                                 padding="valid", trainable=True)
        else:
            raise RuntimeError("Non valid padding type.")

        self.activation_2 = tf.keras.layers.Activation("relu")

        if instance_normalization:
            self.bn_1 = tfda.layers.InstanceNormalization(trainable=True)
            self.bn_2 = tfda.layers.InstanceNormalization(trainable=True)
        else:
            self.bn_1 = tf.keras.layers.BatchNormalization(trainable=True)
            self.bn_2 = tf.keras.layers.BatchNormalization(trainable=True)

        if padding == "default":
            self.conv_2 = tf.keras.layers.Conv2D(filters=self.filter_num, kernel_size=self.filter_size,
                                                 padding="same", trainable=True)

        elif padding == "reflect":
            self.pad_2 = ReflectionPadding2D(padding=(1, 1))
            self.conv_2 = tf.keras.layers.Conv2D(filters=self.filter_num, kernel_size=self.filter_size,
                                                 padding="valid", trainable=True)
        else:
            raise RuntimeError("Non valid padding type.")

        self.activation_3 = tf.keras.layers.Activation("relu")

    def call(self, inputs, *args, **kwargs):

        identity = self.activation_1(inputs)
        x = identity
        if self.padding_type == "reflect":
            x = self.pad_1(x)
        x = self.conv_1(x)
        x = self.activation_2(x)
        x = self.bn_1(x)
        if self.padding_type == "reflect":
            x = self.pad_2(x)
        x = self.conv_2(x)
        x = self.bn_2(x)
        residual = tf.keras.layers.Add()([x, identity])
        x = self.activation_3(residual)
        return x


class ReflectionPadding2D(tf.keras.layers.Layer):

    def __init__(self, padding: Tuple[int, int]):
        super(ReflectionPadding2D, self).__init__()
        self.pad_width, self.pad_height = padding

    def call(self, inputs, *args, **kwargs):
        padding_tensor = tf.constant([
            [0, 0],  # Batch
            [self.pad_height, self.pad_height],  # Height
            [self.pad_width, self.pad_width],  # Width
            [0, 0]  # Channels
        ])

        return tf.pad(inputs, padding_tensor, mode="REFLECT")


if __name__ == "__main__":

    pass
