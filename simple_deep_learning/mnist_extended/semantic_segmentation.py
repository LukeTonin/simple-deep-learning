"""This module contains functions to create the extended MNIST dataset
for semantic segmentation.
"""
import matplotlib.pyplot as plt
from typing import Tuple

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import patches as mpatches

from .array_overlay import overlay_arrays
from .mnist import preprocess_mnist, download_mnist

plt.rcParams['figure.facecolor'] = 'white'


def create_semantic_segmentation_dataset(num_train_samples: int, num_test_samples: int,
                                         image_shape: Tuple[int, int] = (60, 60),
                                         min_num_digits_per_image: int = 2,
                                         max_num_digits_per_image: int = 4,
                                         num_classes: int = 10,
                                         max_iou: float = 0.2,
                                         labels_are_exclusive: bool = False,
                                         target_is_whole_bounding_box: bool = False,
                                         proportion_of_mnist: float = 1.0,
                                         ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create the extended mnist dataset for semantic segmentation.

    Parameters:
        num_train_samples: Number of training samples to generate.
        num_test_samples: Number of test samples to generate.
        image_shape:  The (height, width) of the image.
        min_num_digits_per_image: The minimum number of digits that can be added
            to each output image. The number is randomly selected between min_num_digits_per_image and
            max_num_digits_per_image (included).
        max_num_digits_per_image: The maximum number of digits that can be added
            to each output image. The number is randomly selected between min_num_digits_per_image and
            max_num_digits_per_image (included).
        num_classes: Integer between 1 and 10. Only select images/labels between 0 and num_classes-1.
        max_iou: The maximum allowed IOU (intersection over union) between two overlayed images.
            A lower number means digits will overlap less.
        labels_are_exclusive: If True, each pixel can only belong to one class. If False,
            a pixel can be multiple digits at the same time.
        target_is_whole_bounding_box: If True, the target for each digit is the whole digit's image.
            If False, only the non null pixels of the digit are the target values.
        proportion_of_mnist: The proportion of total mnist images to use when generating this
            dataset. Smaller values will slightly speed up preprocessing (but not much).

        Returns:
            train_x, train_y, test_x, test_y. The input and target values of train and test.
    """

    (train_images, train_labels), (test_images, test_labels) = download_mnist()

    train_images, train_labels = preprocess_mnist(images=train_images, labels=train_labels, proportion=proportion_of_mnist,
                                                  num_classes=num_classes, normalise=True)

    test_images, test_labels = preprocess_mnist(images=test_images, labels=test_labels, proportion=proportion_of_mnist,
                                                num_classes=num_classes, normalise=True)

    train_x, train_y = create_semantic_segmentation_data_from_digits(
        digits=train_images, digit_labels=train_labels,
        num_samples=num_train_samples,
        image_shape=image_shape,
        min_num_digits_per_image=min_num_digits_per_image,
        max_num_digits_per_image=max_num_digits_per_image,
        num_classes=num_classes, max_iou=max_iou,
        labels_are_exclusive=labels_are_exclusive,
        target_is_whole_bounding_box=target_is_whole_bounding_box)

    test_x, test_y = create_semantic_segmentation_data_from_digits(
        digits=test_images, digit_labels=test_labels,
        num_samples=num_test_samples, image_shape=image_shape,
        min_num_digits_per_image=min_num_digits_per_image,
        max_num_digits_per_image=max_num_digits_per_image,
        num_classes=num_classes, max_iou=max_iou,
        labels_are_exclusive=labels_are_exclusive,
        target_is_whole_bounding_box=target_is_whole_bounding_box)

    return train_x, train_y, test_x, test_y


def create_semantic_segmentation_data_from_digits(digits: np.ndarray,
                                                  digit_labels: np.ndarray,
                                                  num_samples: int,
                                                  image_shape: tuple,
                                                  min_num_digits_per_image: int,
                                                  max_num_digits_per_image: int,
                                                  num_classes: int,
                                                  max_iou: float,
                                                  labels_are_exclusive: bool = False,
                                                  target_is_whole_bounding_box: bool = False
                                                  ) -> Tuple[np.ndarray, np.ndarray]:
    """Create the extended MNIST data (either train or test) for semantic segmentation
        from the provided MNIST digits and labels.

    This is used by create_mnist_extended_semantic_segementation_dataset.
    This function is useful directly if one wants to perform additional preprocessing on
    the original mnist digits (e.g resize or warp digits etc.)

    Parameters:
        digits: The MNIST digits (num_images, height, width, 1) 
        digit_labels: The MNIST labels (num_images,)
        image_shape:  The (height, width) of the image.
        min_num_digits_per_image: The minimum number of digits that can be added
            to each output image. The number is randomly selected between min_num_digits_per_image and
            max_num_digits_per_image (included).
        max_num_digits_per_image: The maximum number of digits that can be added
            to each output image. The number is randomly selected between min_num_digits_per_image and
            max_num_digits_per_image (included).
        num_classes: Integer between 1 and 10. Indicating the number of classes used in the dataset.
        max_iou: The maximum allowed IOU (intersection over union) between two overlayed images.
            A lower number means digits will overlap less.
        labels_are_exclusive: If True, each pixel can only belong to one class. If False,
            a pixel can be multiple digits at the same time.
        target_is_whole_bounding_box: If True, the target for each digit is the whole digit's image.
            If False, only the non null pixels of the digit are the target values.

        Returns:
            train_x, train_y, test_x, test_y. The input and target values of train and test.
    """

    input_data = []
    target_data = []

    for _ in range(num_samples):
        num_digits = np.random.randint(
            min_num_digits_per_image, max_num_digits_per_image + 1)

        input_array, arrays_overlayed, labels_overlayed, bounding_boxes_overlayed = overlay_arrays(
            array_shape=image_shape + (1, ),
            input_arrays=digits,
            input_labels=digit_labels,
            max_array_value=1,
            num_input_arrays_to_overlay=num_digits,
            max_iou=max_iou)

        target_array = create_segmentation_target(images=arrays_overlayed,
                                                  labels=labels_overlayed,
                                                  bounding_boxes=bounding_boxes_overlayed,
                                                  image_shape=image_shape,
                                                  num_classes=num_classes,
                                                  labels_are_exclusive=labels_are_exclusive,
                                                  target_is_whole_bounding_box=target_is_whole_bounding_box)

        input_data.append(input_array)
        target_data.append(target_array)

    input_data = np.stack(input_data)
    target_data = np.stack(target_data)

    return input_data, target_data


def create_segmentation_target(images: np.ndarray,
                               labels: np.ndarray,
                               bounding_boxes: np.ndarray,
                               image_shape: tuple,
                               num_classes: int,
                               labels_are_exclusive: bool = False,
                               target_is_whole_bounding_box: bool = False
                               ) -> np.ndarray:
    """Creates the target (aka y value) based on the base images that were overlayed.

    Parameters:
        images: MNIST digits that were overlayed.
        labels: Labels of the digits that were overlayed.
        bounding_boxes: Bounding boxes (wrt output image) of the digits.
        num_classes: Integer between 1 and 10. Indicating the number of classes used in the dataset.
        max_iou: The maximum allowed IOU (intersection over union) between two overlayed images.
            A lower number means digits will overlap less.
        labels_are_exclusive: If True, each pixel can only belong to one class. If False,
            a pixel can be multiple digits at the same time.
        target_is_whole_bounding_box: If True, the target for each digit is the whole digit's image.
            If False, only the non null pixels of the digit are the target values.

    Returns:
        target for a particular input. An ndarray of shape (image_shape, num_classes)

    """
    if len(bounding_boxes) != len(labels) != len(images):
        raise ValueError(
            f'The length of bounding_boxes must be the same as the length of labels. Received shapes: {bounding_boxes.shape}!={labels.shape}')

    target = np.zeros(image_shape + (num_classes,))

    if labels_are_exclusive:
        exclusivity_mask = np.zeros(image_shape)

    for i in range(len(bounding_boxes)):
        label = labels[i]
        xmin, ymin, xmax, ymax = bounding_boxes[i]

        if target_is_whole_bounding_box:
            target[ymin:ymax, xmin:xmax, [label]] = 1
        else:
            max_array_value = max(target[ymin:ymax, xmin:xmax, [label]].max(), images.max())
            target[ymin:ymax, xmin:xmax, [label]] = images[i] + target[ymin:ymax, xmin:xmax, [label]]

            array1 = np.clip(target, a_min=0, a_max=max_array_value, out=target)


        if labels_are_exclusive:
            target[..., label] = np.where(
                exclusivity_mask, 0, target[..., label])
            exclusivity_mask = np.logical_or(
                exclusivity_mask, target[..., label])

    return target


def display_grayscale_array(array: np.ndarray, title: str = '', figsize: tuple = (6, 6)) -> None:
    """Display the grayscale input image.

    Parameters:
        image: This can be either an input digit from MNIST of a input image
            from the extended dataset.
        title: If provided, this will be added as title of the plot.
    """
    plt.figure(figsize=figsize)
    ax = plt.gca()

    ax.imshow(array[..., 0], cmap=plt.cm.binary)
    ax.axes.set_yticks([])
    ax.axes.set_xticks([])

    if title:
        plt.title(title)

    plt.show()


def display_segmented_image(y: np.ndarray, threshold: float = 0.5,
                            input_image: np.ndarray = None,
                            alpha_input_image: float = 0.2,
                            figsize: tuple = (6, 6),
                            title: str = '') -> None:
    """Display segemented image.

    This function displays the image where each class is shown in particular color.
    This is useful for getting a rapid view of the performance of the model
    on a few examples.

    Parameters:
        y: The array containing the prediction.
            Must be of shape (image_shape, num_classes)
        threshold: The threshold used on the predictions.
        input_image: If provided, display the input image in black.
        alpha_input_image: If an input_image is provided, the transparency of
            the input_image.
    """
    base_array = np.ones(
        (y.shape[0], y.shape[1], 3)) * 1
    legend_handles = []

    for i in range(y.shape[-1]):
        # Retrieve a color (without the transparency value).
        colour = plt.cm.jet(i / y.shape[-1])[:-1]
        base_array[y[..., i] > threshold] = colour
        legend_handles.append(mpatches.Patch(color=colour, label=str(i)))

    plt.figure(figsize=figsize)
    plt.imshow(base_array)
    plt.legend(handles=legend_handles, bbox_to_anchor=(1, 1), loc='upper left')
    plt.yticks([])
    plt.xticks([])
    plt.title(title)

    if input_image is not None:
        plt.imshow(input_image[..., 0],
                   cmap=plt.cm.binary, alpha=alpha_input_image)

    plt.show()


def plot_class_masks(y_true: np.ndarray, y_predicted: np.ndarray = None, title='') -> None:
    """Plot a particular view of the true vs predicted segmentation.

    This function separates each class into its own image and
    does not perform any thresholding.

    Parameters:
        y_true: True segmentation (image_shape, num_classes).
        y_predicted: Predicted segmentation (image_shape, num_classes).
            If y_predicted is not provided, only the true values are displayed.
    """
    num_rows = 2 if y_predicted else 1

    num_classes = y_true.shape[-1]
    fig, axes = plt.subplots(num_rows, num_classes, figsize=(num_classes * 4, num_rows * 4))
    axes = axes.reshape(-1, num_classes)

    fig.suptitle(title)

    for label in range(num_classes):
        axes[0, label].imshow(y_true[..., label], cmap=plt.cm.binary)
        axes[0, label].axes.set_yticks([])
        axes[0, label].axes.set_xticks([])

        if label == 0:
            axes[0, label].set_ylabel(f'Target')
        
        if y_predicted:
            if label == 0:
                axes[1, label].set_ylabel(f'Predicted')

            axes[1, label].imshow(y_predicted[..., label], cmap=plt.cm.binary)
            axes[1, label].set_xlabel(f'Label: {label}')
            axes[1, label].axes.set_yticks([])
            axes[1, label].axes.set_xticks([])
        else:
            axes[0, label].set_xlabel(f'Label: {label}')


    plt.show()
