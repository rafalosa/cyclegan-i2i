import tensorflow as tf
from typing import Tuple
import matplotlib.pyplot as plt

OUTPUT_SIZE = (256, 256)


def resize(image):
    return tf.image.resize(image, OUTPUT_SIZE)


def paired_augmentation(image):
    image = tf.image.random_crop(image, size=[2, *OUTPUT_SIZE, 3])
    image = tf.image.random_flip_left_right(image)
    return tf.image.random_flip_up_down(image)


def unpaired_augmentation(image):
    image = tf.image.random_crop(image, size=[*OUTPUT_SIZE, 3])
    image = tf.image.random_flip_left_right(image)
    return tf.image.random_flip_up_down(image)


def train_test_load(input_img_dir: str, target_img_dir: str, val_test_size: float = 0.15, paired: bool = False,
                    augmentation: bool = True, batch_size: int = 32, output_size: Tuple[int, int] = (256, 256)):
    def load(filename: str) -> tf.Tensor:
        image = tf.io.read_file(filename)
        image = tf.io.decode_jpeg(image)
        image = tf.cast(image, tf.float32)
        return (image / 127.5) - 1

    shuffle = not paired

    input_dataset = tf.data.Dataset.list_files(f'{input_img_dir}/*.jpg', shuffle=shuffle)
    input_dataset = input_dataset.cache().map(load, num_parallel_calls=tf.data.AUTOTUNE, name="images")

    target_dataset = tf.data.Dataset.list_files(f'{target_img_dir}/*.jpg', shuffle=shuffle)
    target_dataset = target_dataset.cache().map(load, num_parallel_calls=tf.data.AUTOTUNE, name="images")

    val_test_size = int(len(input_dataset) * val_test_size)

    test_input_dataset = input_dataset.take(val_test_size).batch(batch_size)
    input_dataset = input_dataset.skip(val_test_size)

    val_input_dataset = input_dataset.take(val_test_size).batch(batch_size)
    input_dataset = input_dataset.skip(val_test_size)

    test_target_dataset = target_dataset.take(val_test_size).batch(batch_size)
    target_dataset = target_dataset.skip(val_test_size)

    val_target_dataset = target_dataset.take(val_test_size).batch(batch_size)
    target_dataset = target_dataset.skip(val_test_size)

    if augmentation:
        if paired:
            raise NotImplementedError
        else:
            train_input_dataset = input_dataset.cache().map(unpaired_augmentation, num_parallel_calls=tf.data.AUTOTUNE)
            train_input_dataset = train_input_dataset.concatenate(input_dataset).batch(batch_size)

            train_target_dataset = target_dataset.cache().map(unpaired_augmentation, num_parallel_calls=tf.data.AUTOTUNE)
            train_target_dataset = train_target_dataset.concatenate(target_dataset).batch(batch_size)
    else:
        train_input_dataset = input_dataset.cache().map(resize, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size)
        train_target_dataset = target_dataset.cache().map(resize, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size)

    return (train_input_dataset, test_input_dataset, val_input_dataset, train_target_dataset, test_target_dataset,
           val_target_dataset)


if __name__ == '__main__':
    train_input_dataset, test_input_dataset, val_input_dataset, train_target_dataset, test_target_dataset, val_target_dataset = train_test_load(
        '../processed/300x300/satellite', '../processed/300x300/maps', augmentation=True, paired=False)
    # train_test_load('../processed/300x300/satellite', '../processed/300x300/maps', augmentation=False)
    print(len(train_input_dataset), len(test_input_dataset), len(val_input_dataset))
    plt.subplot(121)
    plt.imshow(next(iter(train_input_dataset))[0] * 0.5 + 0.5)
    plt.subplot(122)
    plt.imshow(next(iter(train_target_dataset))[0] * 0.5 + 0.5)
    plt.show()
