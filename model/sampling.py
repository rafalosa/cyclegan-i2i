import tensorflow as tf
from typing import Optional, Any


def downsampler(filter_num: int, filter_size: int, stride: int = 2, input_shape: Optional[tuple] = None,
                padding: str = "same", apply_bn: bool = True, seed: Any = None, name=None):

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
        result.add(tf.keras.layers.BatchNormalization())

    result.add(tf.keras.layers.LeakyReLU())

    return result


# Można jeszcze wypróbować upsampler z Unpoolingiem i Conv2D
def transposed_upsampler(filter_num: int, filter_size: int, stride: int = 2,
                         padding: str = "same", apply_dropout=False, seed: Any = None, name=None):

    initializer = tf.random_normal_initializer(0., 0.02, seed=seed)

    result = tf.keras.Sequential(name=name)
    result.add(tf.keras.layers.Conv2DTranspose(filter_num, filter_size, strides=stride,
                                               padding=padding,
                                               kernel_initializer=initializer,
                                               use_bias=False))

    result.add(tf.keras.layers.BatchNormalization())

    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))

    result.add(tf.keras.layers.ReLU())

    return result
