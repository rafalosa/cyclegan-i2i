import numpy as np
# import tensorflow as tf
import matplotlib.pyplot as plt
from preprocessing import dataset
from preprocessing import image_preparation

if __name__ == '__main__':

    target = "dataset"
    dataset.get_maps_dataset(target_directory=target, verbose=True)
    image_preparation.split_and_divide(dataset_path="dataset/maps", processed_path="processed", final_image_divisor=2)

