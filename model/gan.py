import tensorflow as tf
from model.generator import UNetGenerator, ResNetGenerator
from model.discriminator import PatchGANDiscriminator
from typing import Any, Literal, Optional
import numpy as np


class GeneratorLoss:

    def __init__(self, lambda_: float = 100):
        self._bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        # self._bce = tf.keras.losses.MeanSquaredError()
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
        # self._bce = tf.keras.losses.MeanSquaredError()
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

    def __init__(self, input_dim: int = 256, seed: Any = None, lambda_: float = 10, upsampling: Literal["deconv", "upconv"] = "deconv",
                 discriminator_loss_buffer_length: int = 50, generator: Literal["unet", "resnet"] = "unet", padding: Literal["default", "reflect"] = "default",
                 residual_blocks: int = 9, resnet_filters: int = 64, discriminator_noise: bool = False, instance_normalization: bool = False):
        super(CycleGAN, self).__init__()

        self.seed: Any = seed

        self.lambda_ = lambda_

        self.discr_noise = discriminator_noise

        self.generator_type = generator

        if self.generator_type == "unet":
            self.t2i_generator: tf.keras.models.Model = UNetGenerator(input_dim=input_dim, seed=seed,
                                                                      upsampling=upsampling,
                                                                      instance_normalization=instance_normalization)
            self.i2t_generator: tf.keras.models.Model = UNetGenerator(input_dim=input_dim, seed=seed,
                                                                      upsampling=upsampling,
                                                                      instance_normalization=instance_normalization)

        elif self.generator_type == "resnet":
            self.t2i_generator: tf.keras.models.Model = ResNetGenerator(input_dim=input_dim, seed=seed,
                                                                        residual_blocks=residual_blocks,
                                                                        filters=resnet_filters, padding_type=padding,
                                                                        instance_normalization=instance_normalization)
            self.i2t_generator: tf.keras.models.Model = ResNetGenerator(input_dim=input_dim, seed=seed,
                                                                        residual_blocks=residual_blocks,
                                                                        filters=resnet_filters, padding_type=padding,
                                                                        instance_normalization=instance_normalization)

            self.t2i_generator.build(input_shape=(None, 256, 256, 3))
            self.i2t_generator.build(input_shape=(None, 256, 256, 3))
        else:
            raise RuntimeError(f"Wrong generator type. Only 'unet' or 'resnet' are valid, got {self.generator_type}")

        self.input_domain_discriminator: tf.keras.models.Model = PatchGANDiscriminator(input_dim=input_dim,
                                                                                       seed=seed, comparing=False,
                                                                                       add_noise=self.discr_noise,
                                                                                       instance_normalization=instance_normalization)
        self.target_domain_discriminator: tf.keras.models.Model = PatchGANDiscriminator(input_dim=input_dim,
                                                                                        seed=seed, comparing=False,
                                                                                        add_noise=self.discr_noise,
                                                                                        instance_normalization=instance_normalization)
        self.generator_loss = GeneratorLoss(lambda_=lambda_)
        self.discriminator_loss = DiscriminatorLoss()
        self.test_metric = tf.keras.metrics.RootMeanSquaredError()

        self.i2t_generator_optimizer = None
        self.input_domain_discriminator_optimizer = None
        self.t2i_generator_optimizer = None
        self.target_domain_discriminator_optimizer = None
        self.optimizers = []

        self.input_discriminator_loss_buffer = []
        self.target_discriminator_loss_buffer = []

        self.buffer_length = discriminator_loss_buffer_length

    def compile(self, input_domain_discriminator_optimizer=None, i2t_generator_optimizer=None,
                target_domain_discriminator_optimizer=None, t2i_generator_optimizer=None, **kwargs):

        self.i2t_generator_optimizer = i2t_generator_optimizer
        self.input_domain_discriminator_optimizer = input_domain_discriminator_optimizer
        self.t2i_generator_optimizer = t2i_generator_optimizer
        self.target_domain_discriminator_optimizer = target_domain_discriminator_optimizer

        self.optimizers = [self.i2t_generator_optimizer,
                           self.input_domain_discriminator_optimizer,
                           self.t2i_generator_optimizer,
                           self.target_domain_discriminator_optimizer]

        super(CycleGAN, self).compile(**kwargs)

    def call(self, inputs, training=None, mask=None):
        raise NotImplemented("CycleGAN call is not implemented, please use generate_i2t and generate_t2i methods"
                             "instead.")

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

            if len(self.input_discriminator_loss_buffer) == self.buffer_length:
                self.input_discriminator_loss_buffer = self.input_discriminator_loss_buffer[1:]
                self.target_discriminator_loss_buffer = self.target_discriminator_loss_buffer[1:]

            self.input_discriminator_loss_buffer.append((input_domain_discriminator_real, input_domain_discriminator_fake))
            self.target_discriminator_loss_buffer.append((target_domain_discriminator_real, target_domain_discriminator_fake))

            i2t_generator_loss, _, _ = self.generator_loss(target_domain_discriminator_fake)
            t2i_generator_loss, _, _ = self.generator_loss(input_domain_discriminator_fake)

            input_domain_cycle_loss = tf.reduce_mean(tf.abs(real_input_domain_image - cycled_input_domain_image)) * self.lambda_
            target_domain_cycle_loss = tf.reduce_mean(tf.abs(real_target_domain_image - cycled_target_domain_image)) * self.lambda_

            input_domain_identity_loss = tf.reduce_mean(tf.abs(real_input_domain_image - input_domain_identity)) * .5 * self.lambda_
            target_domain_identity_loss = tf.reduce_mean(tf.abs(real_target_domain_image - target_domain_identity)) * .5 * self.lambda_

            cycle_loss = input_domain_cycle_loss + target_domain_cycle_loss

            if len(self.input_discriminator_loss_buffer) == self.buffer_length:
                input_domain_discriminator_loss = np.mean([self.discriminator_loss(input_real, input_fake, coefficient=.5) for input_real, input_fake in self.input_discriminator_loss_buffer])
                target_domain_discriminator_loss = np.mean([self.discriminator_loss(target_real, target_fake, coefficient=.5) for target_real, target_fake in self.target_discriminator_loss_buffer])
            else:
                input_domain_discriminator_loss = self.discriminator_loss(input_domain_discriminator_real, input_domain_discriminator_fake, coefficient=.5)
                target_domain_discriminator_loss = self.discriminator_loss(target_domain_discriminator_real, target_domain_discriminator_fake, coefficient=.5)

            # Identity loss helps to retain original colors which is not wanted for our case of i2i translation.
            total_i2t_generator_loss = i2t_generator_loss + cycle_loss + target_domain_identity_loss * 0
            total_t2i_generator_loss = t2i_generator_loss + cycle_loss + input_domain_identity_loss * 0

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
