"""This module contains functions to download and process the mnist dataset.
"""
from typing import Tuple

import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf

plt.rcParams['figure.facecolor'] = 'white'


def download_mnist():
    """Wrapper around keras mnist download function.
    This function uses the keras function. The original data can be found at:
    http://yann.lecun.com/exdb/mnist/
    """

    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

    return (train_images, train_labels), (test_images, test_labels)


def preprocess_mnist(images: np.ndarray, labels: np.ndarray, proportion: float,
                     num_classes: int, normalise: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """Perform simple preprocessing steps on the mnist data.

    Parameters:
        images: MNIST images (num_images, image_height, image_width).
        labels: MNIST labels (num_images,) (must be the same length as images).
        proportion: The proportion of the total dataset (60,000 images) to use.
        num_classes: Integer between 1 and 10. Only select images/labels between 0 and num_classes-1.
        normalise: If True, normalise the data between 0-1, else leave between 0-255.

    Returns:
        images: The preprocessed MNIST images.
        labels: The preprocessed MNIST labels.
    """

    indices = np.random.randint(len(images), size=int(len(images) * proportion))
    images = images[indices]
    labels = labels[indices]

    valid_examples = np.zeros_like(labels)
    for i in range(0, num_classes):
        valid_examples = np.logical_or(labels == i, valid_examples)

    images = images[valid_examples]
    labels = labels[valid_examples]

    if normalise:
        images = images / 255.0

    images = np.expand_dims(images, -1)

    return images, labels


def display_digits(images: np.ndarray, labels: np.ndarray, num_to_display: int = 25, random: bool = True) -> None:
    """Display a random subset of digits from the MNIST dataset.

    Parameters:
        images: MNIST images (num_images, image_height, image_width, 1) or (num_images, image_height, image_width).
        labels: MNIST labels (num_images,) (must be the same length as images).
        num_to_display: Number of images to display.
        random: If True, display the images at random.
    """
    num_columns = 5
    num_rows = int(np.ceil(num_to_display / num_columns))

    plt.figure(figsize=(num_columns * 2, num_rows * 2))

    indices = np.random.randint(
        len(images), size=num_to_display) if random else range(num_to_display)

    for i, index in enumerate(indices):
        ax = plt.subplot(num_rows, num_columns, i+1)
        ax.set_xticks([])
        ax.set_yticks([])

        # imshow takes the input as (x, y) image instead of (x, y, 1) if the image is grayscale or binary.
        if len(images.shape) == 4:
            ax.imshow(images[index, ..., 0], cmap=plt.cm.binary)
        else:
            ax.imshow(images[index], cmap=plt.cm.binary)

        ax.set_xlabel(labels[index])

    plt.show()
