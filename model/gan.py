import tensorflow as tf
import matplotlib.pyplot as plt
from generator import UNetGenerator
from discriminator import PatchGANDiscriminator
from typing import Any, Tuple, Optional
import glob
import numpy as np


class GeneratorLoss:

    def __init__(self, lambda_: float = 100):
        self._bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self._mae = tf.keras.losses.MeanAbsoluteError()
        self.lambda_ = lambda_

    def __call__(self, discriminator_out, generator_out, target):

        gan_loss = self._bce(tf.ones_like(discriminator_out), discriminator_out)

        l1_loss = self._mae(target, generator_out)

        total_loss = gan_loss + self.lambda_ * l1_loss

        return total_loss, gan_loss, l1_loss


class DiscriminatorLoss:

    def __init__(self):
        self._bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self._mae = tf.keras.losses.MeanAbsoluteError()

    def __call__(self, discriminator_real_out, discriminator_generated_out):

        real_loss = self._bce(tf.ones_like(discriminator_real_out), discriminator_real_out)
        generated_loss = self._bce(tf.zeros_like(discriminator_generated_out), discriminator_generated_out)

        total_discriminator_loss = real_loss + generated_loss

        return total_discriminator_loss


class GAN(tf.keras.models.Model):

    def __init__(self, input_dim: int = 300, seed: Any = None):
        super(GAN, self).__init__()

        self.seed: Any = seed

        self.discriminator: tf.keras.models.Model = PatchGANDiscriminator(input_dim=input_dim, seed=seed)
        self.generator: tf.keras.models.Model = UNetGenerator(input_dim=input_dim, seed=seed)

        self.generator_loss = GeneratorLoss(lambda_=100)
        self.discriminator_loss = DiscriminatorLoss()

        self.discriminator_optimizer: Optional[tf.keras.optimizers.Optimizer] = None
        self.generator_optimizer: Optional[tf.keras.optimizers.Optimizer] = None

    def compile(self, discriminator_optimizer=None, generator_optimizer=None, **kwargs):

        self.discriminator_optimizer = discriminator_optimizer
        self.generator_optimizer = generator_optimizer

        super(GAN, self).compile(**kwargs)

    def call(self, inputs, training=None, mask=None):
        pass

    def fit(self, *args, **kwargs):

        assert(self.discriminator_optimizer is not None and self.generator_optimizer is not None)
        super(GAN, self).fit(*args, **kwargs)

    def train_step(self, data):

        input_image, target_image = data

        input_image = tf.concat(input_image, axis=0)  # TODO: Nie wiem czy to tu ma być - do obgadania
        target_image = tf.concat(target_image, axis=0)

        with tf.GradientTape() as generator_tape, tf.GradientTape() as discriminator_tape:
            generator_out = self.generator(input_image, training=True)
            discriminator_out_for_target = self.discriminator([input_image, target_image], training=True)
            discriminator_out_for_generated = self.discriminator([input_image, generator_out], training=True)

            generator_total_loss, generator_gan_loss, generator_L1_loss =\
                self.generator_loss(discriminator_out_for_generated, generator_out, target_image)
            discriminator_loss = self.discriminator_loss(discriminator_out_for_target, discriminator_out_for_generated)

        generator_gradients = generator_tape.gradient(generator_total_loss, self.generator.trainable_variables)
        discriminator_gradients = discriminator_tape.gradient(discriminator_loss, self.discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(generator_gradients, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(discriminator_gradients, self.discriminator.trainable_variables))

        return {"discr_loss": discriminator_loss, "gene_loss": generator_total_loss}


def load_images(path, num):

    result = []

    sat_images_paths = glob.glob(f"{path}/satellite/*.jpg")
    map_images_paths = glob.glob(f"{path}/maps/*.jpg")

    for i, (sat_path, map_path) in enumerate(zip(sat_images_paths, map_images_paths)):

        sat_image = tf.io.read_file(sat_path)
        sat_image = tf.io.decode_jpeg(sat_image)

        map_image = tf.io.read_file(map_path)
        map_image = tf.io.decode_jpeg(map_image)

        result.append((tf.cast(sat_image, tf.float32)[tf.newaxis]/255.0, tf.cast(map_image, tf.float32)[tf.newaxis]/255.0))

        if i >= num:
            break

    return result


if __name__ == "__main__":

    model = GAN(input_dim=300, seed=666)
    model.compile(discriminator_optimizer=tf.keras.optimizers.Adam(2e-4, beta_1=0.5),
                  generator_optimizer=tf.keras.optimizers.Adam(2e-4, beta_1=0.5))

    dataset = load_images("../processed/300x300", num=100)
    sats = [tup[0] for tup in dataset]
    maps = [tup[1] for tup in dataset]

    model.fit(sats, maps, epochs=100)

    sample = plt.imread("../processed/300x300/satellite/0063.jpg")
    map_sample = plt.imread("../processed/300x300/maps/0063.jpg")

    pred = model.generator.predict(sample[tf.newaxis])

    plt.subplot(1, 3, 1)
    plt.imshow(sample)
    plt.subplot(1, 3, 2)
    plt.imshow(map_sample)
    plt.subplot(1, 3, 3)
    plt.imshow(pred.reshape(300, 300, 3))
    plt.show()
