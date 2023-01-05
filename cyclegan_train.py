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


if __name__ == '__main__':

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

    run = wandb.init(project="cycle_gan", entity="golem-rm")

    wandb.config = {
        "lr_target_discr": 2e-4,
        "lr_input_discr": 2e-4,
        "lr_i2t_generator": 1e-4,
        "lr_t2i_generator": 2e-3,
        "epochs": epochs,
        "batch_size": 16,
        "split": split,
        "seed": seed,
        "training_dir": training_dir_name,
        "architecture": "CycleGAN"
    }

    target = "dataset"
    dataset.get_maps_dataset(target_directory=target, verbose=True)
    image_preparation.split_and_divide(dataset_path="dataset/maps", processed_path="processed", final_image_divisor=2)

    model = CycleGAN(input_dim=256, seed=seed)

    model.compile(input_domain_discriminator_optimizer=tf.keras.optimizers.Adam(2e-4, beta_1=0.5),
                  i2t_generator_optimizer=tf.keras.optimizers.Adam(1e-4, beta_1=0.5),
                  target_domain_discriminator_optimizer=tf.keras.optimizers.Adam(2e-4, beta_1=0.5),
                  t2i_generator_optimizer=tf.keras.optimizers.Adam(2e-3, beta_1=0.5))

    checkpoint = tf.train.Checkpoint(cycle_gan=model)

    checkpoint_manager = tf.train.CheckpointManager(checkpoint, training_dir_name, max_to_keep=3)

    if checkpoint_manager.latest_checkpoint:
        checkpoint.restore(checkpoint_manager.latest_checkpoint)
        print("Restored last model checkpoint.")

    (train_input_dataset, test_input_dataset, val_input_dataset, train_target_dataset, test_target_dataset,
     val_target_dataset) = train_test_load(input_img_dir="processed/300x300/satellite",
                                           target_img_dir="processed/300x300/maps",
                                           val_test_size=split, paired=False, augmentation=False)

    (_, test_input_dataset_met, val_input_dataset_met, _, test_target_dataset_met,
     val_target_dataset_met) = train_test_load(input_img_dir="processed/300x300/satellite",
                                               target_img_dir="processed/300x300/maps",
                                               val_test_size=split, paired=True, augmentation=False)

    sample_satellite_images = [
        np.array(plt.imread("processed/300x300/satellite/0014.jpg"))[:256, :256]/127.5 - 1,
        np.array(plt.imread("processed/300x300/satellite/0100.jpg"))[:256, :256]/127.5 - 1,
        np.array(plt.imread("processed/300x300/satellite/0422.jpg"))[:256, :256]/127.5 - 1]

    sample_map_images = [
        np.array(plt.imread("processed/300x300/maps/0014.jpg"))[:256, :256]/127.5 - 1,
        np.array(plt.imread("processed/300x300/maps/0100.jpg"))[:256, :256] / 127.5 - 1,
        np.array(plt.imread("processed/300x300/maps/0422.jpg"))[:256, :256] / 127.5 - 1]

    log_images = (sample_satellite_images, sample_map_images)

    training_dataset = tf.data.Dataset.zip((train_input_dataset, train_target_dataset))
    test_dataset = tf.data.Dataset.zip((test_input_dataset, test_target_dataset))
    val_dataset = tf.data.Dataset.zip((val_input_dataset, val_target_dataset))

    test_dataset_metric = tf.data.Dataset.zip((test_input_dataset_met, test_target_dataset_met))

    history = model.fit(training_dataset, epochs=epochs,
                        batch_size=32,
                        callbacks=[TrainLogger(checkpoint_manager=checkpoint_manager,
                                               image_log=log_images, save_every=5)],
                        validation_data=test_dataset_metric)

    run.finish()
