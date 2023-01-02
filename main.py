import numpy as np
import tensorflow as tf
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from preprocessing import dataset, image_preparation, train_test_load
from utils import TrainLogger
from model import CycleGAN
import argparse
from typing import Any
from datetime import datetime
import os


if __name__ == '__main__':

    target = "dataset"
    dataset.get_maps_dataset(target_directory=target, verbose=True)
    image_preparation.split_and_divide(dataset_path="dataset/maps", processed_path="processed", final_image_divisor=2)

    parser = argparse.ArgumentParser(description='CycleGAN training script.')

    parser.add_argument("--epochs", metavar="e", type=int, default=100,
                        help="number of epochs for model training")

    parser.add_argument("--split", metavar="s", type=float, default=0.15,
                        help="validation dataset relative size")

    parser.add_argument("--seed", metavar="S", type=int, default=666,
                        help="deterministic seed for parameter initializers")

    parser.add_argument("--training_dir", type=str, default=None,
                        help="directory for storing model checkpoints. If not empty, the script will attempt to restore"
                             "last saved checkpoint.")

    args = parser.parse_args()
    epochs = args.epochs
    split = args.split
    seed = args.seed
    dir_candidate = args.training_dir

    if dir_candidate is not None:
        training_dir_name = dir_candidate
    else:
        training_dir_name = f"cycle_gan_training_{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}_e{epochs}"
        os.mkdir(training_dir_name)
        print("Created new training directory.")

    model = CycleGAN(input_dim=256, seed=seed)

    model.compile(input_domain_discriminator_optimizer=Adam(2e-4, beta_1=0.5),
                  i2t_generator_optimizer=Adam(2e-4, beta_1=0.5),
                  target_domain_discriminator_optimizer=Adam(2e-4, beta_1=0.5),
                  t2i_generator_optimizer=Adam(2e-4, beta_1=0.5))

    checkpoint = tf.train.Checkpoint(i2t_generator=model.i2t_generator,
                                     t2i_generator=model.t2i_generator,
                                     input_domain_discriminator=model.input_domain_discriminator,
                                     target_domain_discriminator=model.target_domain_discriminator,
                                     i2t_generator_opti=model.i2t_generator_optimizer,
                                     t2i_generator_opti=model.t2i_generator_optimizer,
                                     input_domain_discriminator_opti=model.input_domain_discriminator_optimizer,
                                     target_domain_discriminator_opti=model.target_domain_discriminator_optimizer)

    checkpoint_manager = tf.train.CheckpointManager(checkpoint, training_dir_name, max_to_keep=3)

    # todo: Figure out how is this exactly supposed to work when I have access to the main machine.
    if checkpoint_manager.latest_checkpoint:
        checkpoint.restore(checkpoint_manager.latest_checkpoint)

    (train_input_dataset, test_input_dataset, val_input_dataset, train_target_dataset, test_target_dataset,
     val_target_dataset) = train_test_load(input_img_dir="../processed/300x300/satellite",
                                           target_img_dir="../processed/300x300/maps",
                                           val_test_size=split, paired=False)

    training_dataset = tf.data.Dataset.zip((train_input_dataset, train_target_dataset))
    test_dataset = tf.data.Dataset.zip((test_input_dataset, test_target_dataset))
    val_dataset = tf.data.Dataset.zip((val_input_dataset, val_target_dataset))

    history = model.fit(training_dataset, epochs=epochs,
                        verbose=0,
                        callbacks=[TrainLogger(checkpoint_manager=checkpoint_manager, save_every=5)],
                        validation_data=val_dataset)
