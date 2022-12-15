import tensorflow as tf
import matplotlib.pyplot as plt
from model.generator import UNetGenerator
from model.discriminator import PatchGANDiscriminator
from typing import Any, Tuple, Optional
import glob
import numpy as np
# import tensorflow_datasets as tfds


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



if __name__ == "__main__":

    BUFFER_SIZE = 100
    BATCH_SIZE = 8

    model = GAN(input_dim=300, seed=666)
    model.compile(discriminator_optimizer=tf.keras.optimizers.Adam(2e-4, beta_1=0.5),
                  generator_optimizer=tf.keras.optimizers.Adam(2e-4, beta_1=0.5))

    train_dataset_sats = tf.data.Dataset.list_files('../processed/300x300/satellite/*.jpg', shuffle=False)
    train_dataset_sats = train_dataset_sats.map(load, num_parallel_calls=tf.data.AUTOTUNE, name="images")
    train_dataset_sats = train_dataset_sats.batch(BATCH_SIZE)

    train_dataset_maps = tf.data.Dataset.list_files('../processed/300x300/maps/*.jpg', shuffle=False)
    train_dataset_maps = train_dataset_maps.map(load, num_parallel_calls=tf.data.AUTOTUNE, name="images")
    train_dataset_maps = train_dataset_maps.batch(BATCH_SIZE)

    dataset = tf.data.Dataset.zip((train_dataset_sats, train_dataset_maps))

    model.fit(dataset, epochs=1)

    sample = plt.imread("../processed/300x300/satellite/0063.jpg")
    pred = model.generator.predict(sample[tf.newaxis], batch_size=1)

    plt.imshow(pred.reshape(300, 300, 3))
    plt.show()
