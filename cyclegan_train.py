import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import wandb
from preprocessing import dataset, image_preparation, train_test_load
from utils import TrainLogger
from model import CycleGAN
import argparse
from datetime import datetime
import os


class CustomLRScheduler(tf.keras.callbacks.Callback):

    def __init__(self, decay_start: int, epochs_till_null: int, initial_learning_rate: float):
        super(CustomLRScheduler, self).__init__()
        self.start_decay = decay_start
        self.initial_lr = initial_learning_rate
        self.epochs_till_zero = epochs_till_null

    def on_epoch_end(self, epoch, logs=None):
        if epoch <= self.start_decay:
            lr = self.initial_lr
        else:
            lr = self.initial_lr * (self.epochs_till_zero - (epoch - self.start_decay))/self.epochs_till_zero

        for optimizer in self.model.optimizers:
            optimizer.lr.assign(lr)


def norm_image(image):
    return np.array(image)[:256, :256] / 127.5 - 1


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='CycleGAN training script.')

    parser.add_argument("--epochs", metavar="e", type=int, default=200,
                        help="number of epochs for model training")

    parser.add_argument("--split", metavar="s", type=float, default=0.15,
                        help="validation dataset relative size")

    parser.add_argument("--seed", metavar="S", type=int, default=666,
                        help="deterministic seed for parameter initializers")

    parser.add_argument("--training_dir", type=str, default=None,
                        help="directory for storing model checkpoints. If not empty, the script will attempt to restore"
                             "last saved checkpoint.")

    parser.add_argument("--upsampling", type=str, default="deconv",
                        help="upsampling mechanism of the generators. Either deconv or upconv.")

    parser.add_argument("--batch_size", type=int, default=128,
                        help="training batch size.")

    parser.add_argument("--loss_buffer", type=int, default=50,
                        help="discriminators loss buffer length which helps to reduce oscillations.")

    parser.add_argument("--init_lr", type=float, default=2e-4,
                        help="initial learning rate for all models.")

    parser.add_argument("--start_decay", type=int, default=100,
                        help="after which epoch start the linear learning rate decay.")

    parser.add_argument("--message", type=str, default=None,
                        help="message to add to the wandb config.")

    parser.add_argument("--generator", type=str, default="unet",
                        help="generator architecture. Either 'unet' or 'resnet'.")

    parser.add_argument("--resnet_padding", type=str, default=None,
                        help="resnet generator padding. Either 'default' or 'reflect'.")

    parser.add_argument("--resnet_filters", type=int, default=None,
                        help="resnet used at the beginning of resnet generator.")

    parser.add_argument("--residual_blocks", type=int, default=None,
                        help="number of residual blocks in resnet generator.")

    parser.add_argument("--discriminator_noise", type=bool, default=False,
                        help="include standard gaussian noise at the discriminator input.")

    parser.add_argument("--instance_norm", type=bool, default=False,
                        help="replace batch normalization layers with instance normalization. Instance normalization"
                             "layers offer greater training stability for a range of smaller batch_sizes with linearly"
                             "adjusted learning rate.")

    args = parser.parse_args()
    epochs = args.epochs
    split = args.split
    seed = args.seed
    dir_candidate = args.training_dir
    upsampling = args.upsampling
    batch_size = args.batch_size
    loss_buffer = args.loss_buffer
    init_lr = args.init_lr
    decay_start = args.start_decay
    message = args.message
    generator = args.generator
    resnet_padding = args.resnet_padding
    residual_blocks = args.residual_blocks
    resnet_filters = args.resnet_filters
    discr_noise = args.discriminator_noise
    instance_norm = args.instance_norm

    if dir_candidate is not None:
        training_dir_name = dir_candidate
    else:
        training_dir_name = f"cycle_gan_training_{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}_e{epochs}"
        os.mkdir(training_dir_name)
        print("Created new training directory.")

    config = {
        "epochs": epochs,
        "batch_size": batch_size,
        "split": split,
        "seed": seed,
        "training_dir": training_dir_name,
        "architecture": "CycleGAN",
        "upsampling_mechanism": upsampling,
        "loss_buffer": loss_buffer,
        "initial_learning_rate": init_lr,
        "start_lr_decay": decay_start,
        "message": message,
        "generator": generator,
        "resnet_padding": resnet_padding,
        "residual_blocks": residual_blocks,
        "resnet_filters": resnet_filters,
        "include_discr_noise": discr_noise
    }

    run = wandb.init(project="cycle_gan", entity="golem-rm", config=config)

    target = "dataset"
    dataset.get_maps_dataset(target_directory=target, verbose=True)
    image_preparation.split_and_divide(dataset_path="dataset/maps", processed_path="processed", final_image_divisor=2)

    model = CycleGAN(input_dim=256, seed=seed, upsampling=upsampling, discriminator_loss_buffer_length=loss_buffer,
                     generator=generator, padding=resnet_padding, residual_blocks=residual_blocks, resnet_filters=resnet_filters,
                     discriminator_noise=discr_noise, instance_normalization=instance_norm)

    model.compile(input_domain_discriminator_optimizer=tf.keras.optimizers.Adam(learning_rate=init_lr, beta_1=0.5),
                  i2t_generator_optimizer=tf.keras.optimizers.Adam(learning_rate=init_lr, beta_1=0.5),
                  target_domain_discriminator_optimizer=tf.keras.optimizers.Adam(learning_rate=init_lr, beta_1=0.5),
                  t2i_generator_optimizer=tf.keras.optimizers.Adam(learning_rate=init_lr, beta_1=0.5))

    checkpoint = tf.train.Checkpoint(cycle_gan=model)

    checkpoint_manager = tf.train.CheckpointManager(checkpoint, training_dir_name, max_to_keep=3)

    if checkpoint_manager.latest_checkpoint:
        checkpoint.restore(checkpoint_manager.latest_checkpoint)
        print("Restored last model checkpoint.")

    (train_input_dataset, test_input_dataset, val_input_dataset, train_target_dataset, test_target_dataset,
     val_target_dataset) = train_test_load(input_img_dir="processed/300x300/satellite",
                                           target_img_dir="processed/300x300/maps",
                                           val_test_size=split, paired=False, augmentation=False,
                                           batch_size=1)

    (_, test_input_dataset_met, val_input_dataset_met, _, test_target_dataset_met,
     val_target_dataset_met) = train_test_load(input_img_dir="processed/300x300/satellite",
                                               target_img_dir="processed/300x300/maps",
                                               val_test_size=split, paired=True, augmentation=False,
                                               batch_size=1)
    sample_satellite_images = [
        norm_image(plt.imread("processed/300x300/satellite/0014.jpg")),
        norm_image(plt.imread("processed/300x300/satellite/0100.jpg")),
        norm_image(plt.imread("processed/300x300/satellite/8318.jpg")),
        norm_image(plt.imread("processed/300x300/satellite/8526.jpg")),
        norm_image(plt.imread("processed/300x300/satellite/8380.jpg")),
        norm_image(plt.imread("processed/300x300/satellite/8538.jpg"))]

    sample_map_images = [
        norm_image(plt.imread("processed/300x300/maps/0014.jpg")),
        norm_image(plt.imread("processed/300x300/maps/0100.jpg")),
        norm_image(plt.imread("processed/300x300/maps/8318.jpg")),
        norm_image(plt.imread("processed/300x300/maps/8526.jpg")),
        norm_image(plt.imread("processed/300x300/maps/8380.jpg")),
        norm_image(plt.imread("processed/300x300/maps/8538.jpg"))]

    log_images = (sample_satellite_images, sample_map_images)

    training_dataset = tf.data.Dataset.zip((train_input_dataset, train_target_dataset))
    test_dataset = tf.data.Dataset.zip((test_input_dataset, test_target_dataset))
    val_dataset = tf.data.Dataset.zip((val_input_dataset, val_target_dataset))

    test_dataset_metric = tf.data.Dataset.zip((test_input_dataset_met, test_target_dataset_met))

    history = model.fit(training_dataset, epochs=epochs,
                        callbacks=[TrainLogger(checkpoint_manager=checkpoint_manager,
                                               image_log=log_images, save_every=5),
                                   CustomLRScheduler(decay_start=decay_start,
                                                     epochs_till_null=epochs-decay_start,
                                                     initial_learning_rate=init_lr)],
                        validation_data=test_dataset_metric)

    run.finish()
