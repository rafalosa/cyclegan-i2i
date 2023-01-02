import tensorflow as tf
import time
import numpy as np
from typing import Callable, Optional


class TrainLogger(tf.keras.callbacks.Callback):

    def __init__(self, checkpoint_manager: tf.train.CheckpointManager, save_every: int = 5):
        super(TrainLogger, self).__init__()
        self.train_start_time = 0
        self.train_end_time = 0
        self.current_epoch_start_time = 0
        self.epoch_times = []
        self.epoch = 0
        self.last_save_epoch = 0
        self.checkpoint_manager = checkpoint_manager
        self.save_interval = save_every

    def on_epoch_begin(self, epoch, logs=None):
        self.current_epoch_start_time = time.time()
        self.epoch = epoch + 1

    def on_epoch_end(self, epoch, logs=None):
        self.epoch_times.append(time.time() - self.current_epoch_start_time)
        epoch_string = f"Current epoch {epoch}, models last saved at epoch {self.last_save_epoch}" \
                       f" : {' '.join([f'{metric}={logs[metric]:.4f}' for metric in logs])}"
        print(f"{epoch_string:<{500}}", end="\r")
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
