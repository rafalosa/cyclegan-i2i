import tensorflow as tf
import time
import numpy as np
from typing import Tuple
import wandb


class TrainLogger(tf.keras.callbacks.Callback):

    def __init__(self, checkpoint_manager: tf.train.CheckpointManager, image_log, save_every: int = 5, cycle_gan: bool = True):
        super(TrainLogger, self).__init__()
        self.is_cyclegan = cycle_gan
        self.train_start_time = 0
        self.train_end_time = 0
        self.current_epoch_start_time = 0
        self.epoch_times = []
        self.epoch = 0
        self.last_save_epoch = 0
        self.checkpoint_manager = checkpoint_manager
        self.save_interval = save_every
        self.image_data = image_log

    def on_epoch_begin(self, epoch, logs=None):
        self.current_epoch_start_time = time.time()
        self.epoch = epoch + 1

    def on_epoch_end(self, epoch, logs=None):
        epoch_time = time.time() - self.current_epoch_start_time
        self.epoch_times.append(epoch_time)

        if self.is_cyclegan:

            i2t_images = [((self.model.i2t_generator.predict(image[tf.newaxis], verbose=0).reshape((256, 256, 3)) + 1) * 127.5).astype(int) for image in self.image_data[0]]
            t2i_images = [((self.model.t2i_generator.predict(image[tf.newaxis], verbose=0).reshape((256, 256, 3)) + 1) * 127.5).astype(int) for image in self.image_data[1]]

            wandb.log({
                "input_domain_gt": [wandb.Image(image) for image in self.image_data[0]],
                "target_domain_generated": [wandb.Image(i2t_image) for i2t_image in i2t_images],
                "target_domain_gt": [wandb.Image(image) for image in self.image_data[1]],
                "input_domain_generated": [wandb.Image(t2i_image) for t2i_image in t2i_images],
                **logs
            })

        else:
            t_images = [((self.model.generator.predict(image[tf.newaxis], verbose=0).reshape((256, 256, 3)) + 1) * 127.5).astype(int) for image in self.image_data[0]]

            wandb.log({
                "input_gt": [wandb.Image(image) for image in self.image_data[0]],
                "target_domain_generated": [wandb.Image(t_image) for t_image in t_images],
                **logs
            })

        if (epoch + 1) % self.save_interval == 0:
            self.checkpoint_manager.save()
            self.last_save_epoch = epoch

    def on_train_begin(self, logs=None):
        self.train_start_time = time.time()

    def on_train_end(self, logs=None):
        self.train_end_time = time.time()
        print(f"\nTraining took {self.train_end_time - self.train_start_time:.2f}s,"
              f" with {np.mean(self.epoch_times):.2f}s per epoch on average"
              f"\nTraining ended on epoch {self.epoch}")
