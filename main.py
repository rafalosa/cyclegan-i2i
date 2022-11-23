import numpy as np
# import tensorflow as tf
import matplotlib.pyplot as plt
from preprocessing import dataset

if __name__ == '__main__':

    target = "dataset"
    dataset.get_maps_dataset(target_directory=target, verbose=True)

