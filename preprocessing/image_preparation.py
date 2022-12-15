import os
import glob
import shutil
import matplotlib.pyplot as plt
import numpy as np
from typing import List
import tqdm


def _merge_and_copy(dataset_path: str, processed_path: str):

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

    for i, image_path in enumerate(tqdm.tqdm(img_glob, desc="Copying original images")):
        shutil.copyfile(image_path, os.path.join(processed_path, "all_samples", f"{i:04d}.jpg"))


def _split_combined_images(images_path: str, div=2):
    img_glob = glob.glob(os.path.join(images_path, "all_samples", "*.jpg"))

    maps_path = os.path.join(images_path, "maps")
    satellite_path = os.path.join(images_path, "satellite")

    if "satellite" not in os.listdir(images_path):
        os.mkdir(satellite_path)

    if "maps" not in os.listdir(images_path):
        os.mkdir(maps_path)

    check_glob_sat = glob.glob(os.path.join(satellite_path, "*.jpg"))
    check_glob_map = glob.glob(os.path.join(maps_path, "*.jpg"))

    if check_glob_sat or check_glob_map:
        return

    for i, img_path in enumerate(tqdm.tqdm(img_glob, desc="Splitting original images")):
        img = np.array(plt.imread(img_path))
        satellite_image, map_image = np.split(img, div, axis=1)

        plt.imsave(os.path.join(satellite_path, f"{i:04d}.jpg"), satellite_image)
        plt.imsave(os.path.join(maps_path, f"{i:04d}.jpg"), map_image)


def _divide_images(images_path: str, divisor=2):

    def load_and_split(img_list: List[str], output: str, message: str):

        imgs_num = len(img_list) * divisor**2

        index = 0

        with tqdm.tqdm(total=imgs_num, desc=message) as progress_bar:

            for image in img_list:
                img = np.array(plt.imread(image))
                cols = np.split(img, divisor, axis=1)
                for col in cols:
                    images_from_column = np.split(col, divisor, axis=0)
                    for square_image in images_from_column:
                        plt.imsave(os.path.join(output, f"{index:04d}.jpg"), square_image)
                        progress_bar.update(1)
                        index += 1

    result_dir = f"{600//divisor}x{600//divisor}"
    maps_dir = os.path.join(images_path, result_dir, "maps")
    satellite_dir = os.path.join(images_path, result_dir, "satellite")

    if result_dir not in os.listdir(images_path):
        os.mkdir(os.path.join(images_path, result_dir))

    else:
        return

    if "satellite" not in os.listdir(os.path.join(images_path, result_dir)):
        os.mkdir(satellite_dir)

    if "maps" not in os.listdir(os.path.join(images_path, result_dir)):
        os.mkdir(maps_dir)

    maps_glob = glob.glob(os.path.join(images_path, "maps", "*.jpg"))
    satellite_glob = glob.glob(os.path.join(images_path, "satellite", "*.jpg"))

    load_and_split(maps_glob, maps_dir, "Dividing map images")
    load_and_split(satellite_glob, satellite_dir, "Dividing satellite images")


def split_and_divide(dataset_path: str, processed_path: str, final_image_divisor: int):

    _merge_and_copy(dataset_path, processed_path)
    _split_combined_images(processed_path)
    _divide_images(processed_path, divisor=final_image_divisor)
