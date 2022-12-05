import numpy as np
# import tensorflow as tf
import matplotlib.pyplot as plt
from preprocessing import dataset
from preprocessing import processing

if __name__ == '__main__':

    target = "dataset"
    dataset.get_maps_dataset(target_directory=target, verbose=True)
    processing.split_and_divide(dataset_path="dataset/maps", processed_path="processed", final_image_divisor=2)

