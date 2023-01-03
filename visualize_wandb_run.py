import wandb
from PIL import Image
import numpy as np
import pandas as pd

DURATION = 10
JOULE_TO_KWH = 3600 * 1000


def generate_gif(filenames, gif_name):

    filenames.sort(key=lambda x: int(x.split('_')[-2]))
    frames = [Image.open(image) for image in filenames]
    frame_one = frames[0]
    frame_one.save(f'{gif_name}.gif', format="GIF", append_images=frames,
                   save_all=True, duration=DURATION, loop=0)


def get_energy_for_all_runs(wandb_api):
    runs = wandb_api.runs("golem-rm/cycle_gan")

    energy = 0

    for run in runs:

        mets = run.history(stream="events")

        df = mets[["_timestamp", "system.gpu.process.3.powerWatts", "system.gpu.process.4.powerWatts"]]
        df.dropna(inplace=True)

        tim = np.array(df["_timestamp"])
        ene1 = np.array(df["system.gpu.process.3.powerWatts"])
        ene2 = np.array(df["system.gpu.process.4.powerWatts"])

        energy += np.trapz(ene1, tim) + np.trapz(ene2, tim)

    return energy / JOULE_TO_KWH


if __name__ == "__main__":

    api = wandb.Api()

    print(get_energy_for_all_runs(api))

    # images_fnames = []
    #
    # key = "target_domain_generated"
    #
    # images_fnames.append([el[key]['filenames'][0] for el in hist])
    # images_fnames.append([el[key]['filenames'][1] for el in hist])
    # images_fnames.append([el[key]['filenames'][2] for el in hist])
    #
    # gif_names = ["unet_1_succ_img2", "unet_1_succ_img3"]
    #
    # for gif_name, fnames in zip(gif_names, images_fnames):
    #
    #     for file in run.files():
    #         if file.name in fnames:
    #             file.download(replace=True)
    #
    #     generate_gif(fnames, gif_name)


