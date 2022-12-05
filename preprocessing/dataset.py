import wget
import os
import tarfile
import glob
import tqdm


def get_maps_dataset(target_directory: str, verbose=True):

    def unpack_dataset(path_raw: str, path_unpack: str):

        data_file = tarfile.open(path_raw)
        data_file.extractall(path_unpack)
        data_file.close()

    def create_progress_bar_update(bar: tqdm.tqdm):
        def progress_bar(current, total, *args):
            if bar.total != total:
                bar.total = total
                print("\r", end="")

            bar.update(current - bar.n)

        return progress_bar

    dataset_dir = target_directory
    dataset_temp_dir = ".raw_data"

    dataset_url = "http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/maps.tar.gz"
    dataset_temp_name = "dataset.tar.gz"

    if dataset_temp_dir not in os.listdir():
        os.mkdir(dataset_temp_dir)

    if dataset_temp_name not in os.listdir(dataset_temp_dir):
        if verbose:
            bar_ = tqdm.tqdm(desc="Downloading dataset")
            wget.download(dataset_url, os.path.join(dataset_temp_dir, dataset_temp_name),
                          bar=create_progress_bar_update(bar_))
        else:
            wget.download(dataset_url, os.path.join(dataset_temp_dir, dataset_temp_name), bar=None)

        if verbose:
            print("")

    else:
        pattern = os.path.join(dataset_dir, "**", "*.jpg")
        glb = glob.glob(pattern, recursive=True)
        if glb:
            return
        if verbose:
            print("Using cached dataset archive")

    if dataset_dir not in os.listdir():
        os.mkdir(dataset_dir)
        unpack_dataset(os.path.join(dataset_temp_dir, dataset_temp_name), dataset_dir)
    else:
        if any(os.listdir(dataset_dir)):
            response = input("Target directory exists and is not empty, are you sure you"
                             " want to unpack the dataset to that location? [y/n]").lower()
            if response not in ["y", "n"]:
                while 1:
                    response = input("Please respond with: [y/n]").lower()
                    if response in ["y", "n"]:
                        break

            if response == "y":
                unpack_dataset(os.path.join(dataset_temp_dir, dataset_temp_name), dataset_dir)
            else:
                raise RuntimeError("Specify different target dataset directory.")
        else:
            unpack_dataset(os.path.join(dataset_temp_dir, dataset_temp_name), dataset_dir)




