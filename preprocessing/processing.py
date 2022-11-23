import os
import glob
import shutil
import matplotlib.pyplot as plt
import numpy as np


def merge_and_copy(dataset_path: str, processed_path: str):

    if not ("train" in os.listdir(dataset_path) and "val" not in os.listdir()):
        raise RuntimeError("Dataset path should contain train and val directories.")

    if processed_path not in os.listdir():
        os.mkdir(processed_path)

    if "all_samples" not in os.listdir(processed_path):
        os.mkdir(os.path.join(processed_path, "all_samples"))

    check_glob = glob.glob(os.path.join(processed_path, "all_samples", "*.jpg"))

    if check_glob:
        return

    img_glob = glob.glob(os.path.join(dataset_path, "**", "*.jpg"), recursive=True)

    for i, image_path in enumerate(img_glob):
        shutil.copyfile(image_path, os.path.join(processed_path, "all_samples", f"{i:04d}.jpg"))


def split_combined_images(images_path: str, div=2):
    img_glob = glob.glob(os.path.join(images_path, "all_samples", "*.jpg"))

    maps_path = os.path.join(images_path, "maps")
    satellite_path = os.path.join(images_path, "satellite")

    if "satellite" not in os.listdir(images_path):
        os.mkdir(satellite_path)

    if "maps" not in os.listdir(images_path):
        os.mkdir(maps_path)

    for i, img_path in enumerate(img_glob):
        img = np.array(plt.imread(img_path))
        satellite_image, map_image = np.split(img, div, axis=1)

        plt.imsave(os.path.join(satellite_path, f"{i:04d}.jpg"), satellite_image)
        plt.imsave(os.path.join(maps_path, f"{i:04d}.jpg"), map_image)


def divide_images(images_path: str, divisor=2):
    pass


if __name__ == "__main__":

    split_combined_images("../processed")

