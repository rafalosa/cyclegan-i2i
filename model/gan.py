import tensorflow as tf
import matplotlib.pyplot as plt
from model.generator import UNetGenerator
from model.discriminator import PatchGANDiscriminator
from typing import Any, Tuple, Optional
from preprocessing import train_test_load
import glob
import numpy as np
# import tensorflow_datasets as tfds
import wandb


class GeneratorLoss:

    def __init__(self, lambda_: float = 100):
        self._bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self._mae = tf.keras.losses.MeanAbsoluteError()
        self.lambda_ = lambda_

    def __call__(self, discriminator_out, generator_out=None, target=None):

        gan_loss = self._bce(tf.ones_like(discriminator_out), discriminator_out)

        if generator_out is not None and target is not None:
            l1_loss = self._mae(target, generator_out)
        else:
            l1_loss = 0

        total_loss = gan_loss + self.lambda_ * l1_loss

        return total_loss, gan_loss, l1_loss


class DiscriminatorLoss:

    def __init__(self):
        self._bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self._mae = tf.keras.losses.MeanAbsoluteError()

    def __call__(self, discriminator_real_out, discriminator_generated_out, coefficient: float = 1):

        real_loss = self._bce(tf.ones_like(discriminator_real_out), discriminator_real_out)
        generated_loss = self._bce(tf.zeros_like(discriminator_generated_out), discriminator_generated_out)

        total_discriminator_loss = real_loss + generated_loss

        return total_discriminator_loss * coefficient


class GAN(tf.keras.models.Model):

    def __init__(self, input_dim: int = 300, seed: Any = None):
        super(GAN, self).__init__()

        self.seed: Any = seed

        self.discriminator: tf.keras.models.Model = PatchGANDiscriminator(input_dim=input_dim, seed=seed)
        self.generator: tf.keras.models.Model = UNetGenerator(input_dim=input_dim, seed=seed)

        self.generator_loss = GeneratorLoss()
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


class CycleGAN(tf.keras.models.Model):

    def __init__(self, input_dim: int = 256, seed: Any = None, lambda_: float = 10):
        super(CycleGAN, self).__init__()

        self.seed: Any = seed

        self.lambda_ = lambda_

        self.input_domain_discriminator: tf.keras.models.Model = PatchGANDiscriminator(input_dim=input_dim,
                                                                                       seed=seed, comparing=False)
        self.i2t_generator: tf.keras.models.Model = UNetGenerator(input_dim=input_dim, seed=seed)

        self.target_domain_discriminator: tf.keras.models.Model = PatchGANDiscriminator(input_dim=input_dim,
                                                                                        seed=seed, comparing=False)
        self.t2i_generator: tf.keras.models.Model = UNetGenerator(input_dim=input_dim, seed=seed)

        self.generator_loss = GeneratorLoss(lambda_=lambda_)
        self.discriminator_loss = DiscriminatorLoss()
        self.test_metric = tf.keras.metrics.RootMeanSquaredError()

        self.i2t_generator_optimizer = None
        self.input_domain_discriminator_optimizer = None
        self.t2i_generator_optimizer = None
        self.target_domain_discriminator_optimizer = None

    def compile(self, input_domain_discriminator_optimizer=None, i2t_generator_optimizer=None,
                target_domain_discriminator_optimizer=None, t2i_generator_optimizer=None, **kwargs):

        self.i2t_generator_optimizer = i2t_generator_optimizer
        self.input_domain_discriminator_optimizer = input_domain_discriminator_optimizer
        self.t2i_generator_optimizer = t2i_generator_optimizer
        self.target_domain_discriminator_optimizer = target_domain_discriminator_optimizer

        super(CycleGAN, self).compile(**kwargs)

    def call(self, inputs, training=None, mask=None):
        pass

    # def fit(self, *args, **kwargs):
    #     super(CycleGAN, self).fit(*args, **kwargs)

    def train_step(self, data):
        real_input_domain_image, real_target_domain_image = data

        with tf.GradientTape(persistent=True) as tape:

            fake_target_domain_image = self.i2t_generator(real_input_domain_image, training=True)
            cycled_input_domain_image = self.t2i_generator(fake_target_domain_image, training=True)

            fake_input_domain_image = self.t2i_generator(real_target_domain_image, training=True)
            cycled_target_domain_image = self.i2t_generator(fake_input_domain_image, training=True)

            input_domain_identity = self.t2i_generator(real_input_domain_image, training=True)
            target_domain_identity = self.i2t_generator(real_target_domain_image, training=True)

            input_domain_discriminator_real = self.input_domain_discriminator(real_input_domain_image, training=True)
            target_domain_discriminator_real = self.target_domain_discriminator(real_target_domain_image, training=True)

            input_domain_discriminator_fake = self.input_domain_discriminator(fake_input_domain_image, training=True)
            target_domain_discriminator_fake = self.target_domain_discriminator(fake_target_domain_image, training=True)

            i2t_generator_loss, _, _ = self.generator_loss(target_domain_discriminator_fake)
            t2i_generator_loss, _, _ = self.generator_loss(input_domain_discriminator_fake)

            input_domain_cycle_loss = tf.reduce_mean(tf.abs(real_input_domain_image - cycled_input_domain_image)) * self.lambda_
            target_domain_cycle_loss = tf.reduce_mean(tf.abs(real_target_domain_image - cycled_target_domain_image)) * self.lambda_

            input_domain_identity_loss = tf.reduce_mean(tf.abs(real_input_domain_image - input_domain_identity)) * .5 * self.lambda_
            target_domain_identity_loss = tf.reduce_mean(tf.abs(real_target_domain_image - target_domain_identity)) * .5 * self.lambda_

            cycle_loss = input_domain_cycle_loss + target_domain_cycle_loss

            total_i2t_generator_loss = i2t_generator_loss + cycle_loss + target_domain_identity_loss
            total_t2i_generator_loss = t2i_generator_loss + cycle_loss + input_domain_identity_loss

            input_domain_discriminator_loss = self.discriminator_loss(input_domain_discriminator_real, input_domain_discriminator_fake, coefficient=.5)
            target_domain_discriminator_loss = self.discriminator_loss(target_domain_discriminator_real, target_domain_discriminator_fake, coefficient=.5)

        i2t_generator_grads = tape.gradient(total_i2t_generator_loss,
                                            self.i2t_generator.trainable_variables)
        t2i_generator_grads = tape.gradient(total_t2i_generator_loss,
                                            self.t2i_generator.trainable_variables)
        input_domain_discriminator_grads = tape.gradient(input_domain_discriminator_loss,
                                                         self.input_domain_discriminator.trainable_variables)
        target_domain_discriminator_grads = tape.gradient(target_domain_discriminator_loss,
                                                          self.target_domain_discriminator.trainable_variables)

        self.i2t_generator_optimizer.apply_gradients(zip(i2t_generator_grads, self.i2t_generator.trainable_variables))
        self.t2i_generator_optimizer.apply_gradients(zip(t2i_generator_grads, self.t2i_generator.trainable_variables))

        self.input_domain_discriminator_optimizer.apply_gradients(zip(input_domain_discriminator_grads,
                                                                      self.input_domain_discriminator.trainable_variables))
        self.target_domain_discriminator_optimizer.apply_gradients(zip(target_domain_discriminator_grads,
                                                                       self.target_domain_discriminator.trainable_variables))

        return {"input_discr_loss": input_domain_discriminator_loss, "i2t_gene_loss": total_i2t_generator_loss,
                "target_discr_loss": target_domain_discriminator_loss, "t2i_gene_loss": total_t2i_generator_loss}

    def test_step(self, data):

        # for paired images
        input_domain_ground_truth, target_domain_ground_truth = data

        target_domain_generated = self.i2t_generator(input_domain_ground_truth, training=False)
        input_domain_generated = self.t2i_generator(target_domain_ground_truth, training=False)

        target_domain_metric = self.test_metric(target_domain_ground_truth, target_domain_generated)
        input_domain_metric = self.test_metric(input_domain_ground_truth, input_domain_generated)

        return {"cumulative_generator_error": target_domain_metric+input_domain_metric}

    def generate_i2t(self, image):
        return self.i2t_generator(image)

    def generate_t2i(self, image):
        return self.t2i_generator(image)


if __name__ == "__main__":
    pass
    # import wandb
    #
    # wandb.init(project="test-project", entity="golem-rm")
    #
    # wandb.config = {
    #     "learning_rate": 2e-4,
    #     "epochs": 2,
    #     "batch_size": 8,
    #     "split": 0.2
    # }
    #
    # with tf.compat.v1.Session() as sess:
    #
    #     model = GAN(input_dim=256, seed=666)
    #     model.compile(discriminator_optimizer=tf.keras.optimizers.Adam(2e-4, beta_1=0.5),
    #                   generator_optimizer=tf.keras.optimizers.Adam(2e-4, beta_1=0.5))
    #
    #     (train_input_dataset, test_input_dataset, val_input_dataset, train_target_dataset, test_target_dataset,
    #      val_target_dataset) = train_test_load(input_img_dir="../processed/300x300/satellite", target_img_dir="../processed/300x300/maps",
    #                                            val_test_size=.2, paired=True, augmentation=False)
    #
    #     training_dataset = tf.data.Dataset.zip((train_input_dataset, train_target_dataset))
    #
    #     model.fit(training_dataset, epochs=2)
    #     #
    #     # sample = plt.imread("../processed/300x300/satellite/0063.jpg")
    #     # pred = model.generator.predict(sample[tf.newaxis], batch_size=1)
