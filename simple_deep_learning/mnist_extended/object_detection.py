"""This module contains functions to create the extended MNIST dataset
for object detection.
"""

from typing import Tuple, List, Union

import numpy as np
import PIL
from PIL import ImageDraw, ImageFont

from .array_overlay import overlay_arrays
from .mnist import preprocess_mnist, download_mnist


def create_object_detection_dataset(num_train_samples: int, num_test_samples: int,
                                    image_shape: Tuple[int, int] = (60, 60),
                                    min_num_digits_per_image: int = 2,
                                    max_num_digits_per_image: int = 4,
                                    num_classes: int = 10,
                                    max_iou: float = 0.2,
                                    proportion_of_mnist: float = 1.0,
                                    ) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray],
                                               np.ndarray, List[np.ndarray], List[np.ndarray]]:
    """Create the extended mnist dataset for object detection. 

    The bounding boxes are returned in a format that is not readily usable by 
    a machine learning algorithm. This is because the actual target array used
    in backpropagation will vary depending on the characteristics of the model used (e.g
    number of anchors used, number of feature maps etc...). It is left up to the
    user of this function to process the provided bounding boxes into the array format required
    by the model.

    Parameters:
        num_train_samples: Number of training samples to generate.
        num_test_samples: Number of test samples to generate.
        image_shape: The (height, width) of the image.
        min_num_digits_per_image: The minimum number of digits that can be added
            to each output image. The number is randomly selected between min_num_digits_per_image and
            max_num_digits_per_image (included).
        max_num_digits_per_image: The maximum number of digits that can be added
            to each output image. The number is randomly selected between min_num_digits_per_image and
            max_num_digits_per_image (included).
        num_classes: Integer between 1 and 10. Only select images/labels between 0 and num_classes-1.
        max_iou: The maximum allowed IOU (intersection over union) between two overlaid images.
            A lower number means digits will overlap less.
        proportion_of_mnist: The proportion of total mnist images to use when generating this
            dataset. Smaller values will slightly speed up preprocessing (but not much).

        Returns:
            train_x, train_bounding_boxes, train_labels, test_x, test_bounding_boxes, test_labels.
            The input and the bounding boxes and labels for train and test.
            The bounding boxes are in absolute pixel values in the format xmin, ymin, xmax, ymax
    """

    (train_images, train_labels), (test_images, test_labels) = download_mnist()

    train_images, train_labels = preprocess_mnist(images=train_images, labels=train_labels, proportion=proportion_of_mnist,
                                                  num_classes=num_classes, normalise=True)

    test_images, test_labels = preprocess_mnist(images=test_images, labels=test_labels, proportion=proportion_of_mnist,
                                                num_classes=num_classes, normalise=True)

    train_x, train_bounding_boxes, train_labels = create_object_detection_data_from_digits(
        digits=train_images, digit_labels=train_labels,
        num_samples=num_train_samples, image_shape=image_shape,
        min_num_digits_per_image=min_num_digits_per_image,
        max_num_digits_per_image=max_num_digits_per_image,
        max_iou=max_iou)

    test_x, test_bounding_boxes, test_labels = create_object_detection_data_from_digits(
        digits=test_images, digit_labels=test_labels,
        num_samples=num_test_samples, image_shape=image_shape,
        min_num_digits_per_image=min_num_digits_per_image,
        max_num_digits_per_image=max_num_digits_per_image,
        max_iou=max_iou)

    return train_x, train_bounding_boxes, train_labels, test_x, test_bounding_boxes, test_labels


def create_object_detection_data_from_digits(digits: np.ndarray,
                                             digit_labels: np.ndarray,
                                             num_samples: int,
                                             image_shape: tuple,
                                             min_num_digits_per_image: int,
                                             max_num_digits_per_image: int,
                                             max_iou: float,
                                             ) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray]]:
    """Create the extended MNIST data for object detection from the provided MNIST digits and labels.

    This is used by create_mnist_extended_object_detection_dataset.
    This function is useful directly if one wants to perform additional preprocessing on
    the original mnist digits (e.g resize or warp digits etc.)

    Parameters:
        digits: The MNIST digits (num_images, height, width, 1) 
        labels: The MNIST labels (num_images,)
        image_shape:  The (height, width) of the image.
        min_num_digits_per_image: The minimum number of digits that can be added
            to each output image. The number is randomly selected between min_num_digits_per_image and
            max_num_digits_per_image (included).
        max_num_digits_per_image: The maximum number of digits that can be added
            to each output image. The number is randomly selected between min_num_digits_per_image and
            max_num_digits_per_image (included).
        num_classes: Integer between 1 and 10. Indicating the number of classes used in the dataset.
        max_iou: The maximum allowed IOU (intersection over union) between two overlaid images.
            A lower number means digits will overlap less.

        Returns:
            x, bounding_boxes, labels.
            The input, the bounding boxes and the labels.
            The bounding boxes are absolute pixel values in the format xmin, ymin, xmax, ymax
    """

    x = []
    labels = []
    bounding_boxes = []

    for _ in range(num_samples):
        num_digits = np.random.randint(
            min_num_digits_per_image, max_num_digits_per_image + 1)

        input_array, arrays_overlaid, labels_overlaid, bounding_boxes_overlaid = overlay_arrays(
            array_shape=image_shape + (1, ),
            input_arrays=digits,
            input_labels=digit_labels,
            max_array_value=1,
            num_input_arrays_to_overlay=num_digits,
            max_iou=max_iou)

        x.append(input_array)
        labels.append(labels_overlaid)
        bounding_boxes.append(bounding_boxes_overlaid)

    x = np.stack(x)

    return x, bounding_boxes, labels


def draw_bounding_boxes(image: Union[PIL.Image.Image, np.ndarray],
                        bounding_boxes: Union[list, np.ndarray],
                        labels: Union[list, np.ndarray] = None,
                        label_size: int=7,
                        colour: str = 'white', width: int = 1,
                        copy: bool = False) -> PIL.Image.Image:
    """Draw multiple bounding boxes with labels on an image.

    Essentially a loop over the draw_bounding_box function.

    Parameters:
        See draw_bounding_box for most parameters:
        bounding_boxes: Must indexable with each index returning a bounding box in
            format used by draw_bounding_box.
        labels: Same as bounding_boxes

    Returns:
        A PIL image with the bounding box drawn.
    """

    num_bounding_boxes = len(bounding_boxes)

    if isinstance(image, np.ndarray):
        image = array_to_image(image)
    elif isinstance(image, PIL.Image.Image):
        if copy:
            image = image.copy()
    else:
        raise ValueError(
            f'Invalid type {type(image)} for image argument. Expecting np.ndarray or PIL.Image.Image')

    if labels is not None:
        num_labels = len(labels)
        if num_bounding_boxes != num_labels:
            raise ValueError('len(bounding_boxes) and len(labels) must be the same. '
                             f'len(bounding_boxes) = {num_bounding_boxes} != len(labels) = {num_labels}')

    for i in range(num_bounding_boxes):
        draw_bounding_box(image, bounding_box=bounding_boxes[i], label=labels[i],
                          label_size=label_size,
                          colour=colour, width=width, copy=False)

    return image


def draw_bounding_box(image: Union[PIL.Image.Image, np.ndarray], bounding_box: Union[tuple, list, np.ndarray],
                      label: str = None, label_size: int = 7,
                      colour: str = 'white', width: int = 1, copy: bool = False) -> PIL.Image.Image:
    """Draw a bounding box with a label on an Image.

    Parameters:
        image: A PIL image or numpy array on which to draw the bounding box.
        bounding_box: A tuple/array/list of 4 values containing the 
            xmin, ymin, xmax, ymax coordinates of the bounding box.
        label: A string (or integer) to display next to the bounding box.
        label_size: The size of the label when displayed.
            This is only used if the function manages to load the specified font.
            Otherwise PIL will use the default font for which the size cannot be set.
        colour: The colour of the label and bounding box.
        width: The width of the bounding box.
        copy: If True, copy the image and do not modify the original.
            Only used if image is a PIL Image. In the case of an array,
            the array is copied automatically.

    Returns:
        A PIL image with the bounding box drawn.
    """

    if isinstance(image, np.ndarray):
        image = array_to_image(image)
    elif isinstance(image, PIL.Image.Image):
        if copy:
            image = image.copy()
    else:
        raise ValueError(
            f'Invalid type {type(image)} for image argument. Expecting np.ndarray or PIL.Image.Image')

    xmin, ymin, xmax, ymax = bounding_box

    image_draw = ImageDraw.Draw(image)
    image_draw.rectangle(xy=(xmin, ymin, xmax, ymax),
                         outline=colour, width=width)

    if label is not None:
        label = str(label)

        try:
            font = ImageFont.truetype("arial.ttf", size=label_size)
        except OSError:
            font = ImageFont.load_default()

        label_width, label_height = font.getsize(label)
        image_height, image_width = image.size

        if ymax + label_height > image_height:
            label_x = xmin
            label_y = ymin - label_height
        else:
            label_x = xmin
            label_y = ymax

        image_draw.text((label_x, label_y), label, fill=colour, font=font)

    return image


def array_to_image(array: np.ndarray) -> PIL.Image.Image:
    """Converts an array to a PIL image.

    Performs checks and applies type modifications to
    put into the correct format for PIL.

    Parameters:
        array: An array to convert. Can be have any of the following shape:
            (height, width), (height, width, 1), (height, width, 3)

    Returns:
        A PIL image of mode L or RGB.
    """
    array_max = array.max()
    array_min = array.min()

    if array_max > 255 or array_min < 0:
        raise ValueError('This function cannot deal with values above 255 or '
                         f'below 0. array.max() = {array_max}, array_min = {array_min}')

    if array.dtype == float:
        if array.max() <= 1:
            array = (array * 255).astype(np.uint8)
        else:
            array = array.astype(np.uint8)

    if len(array.shape) == 3:
        if array.shape[2] == 1:
            array = array[..., 0]
        else:
            if array.shape[2] != 3:
                raise ValueError(
                    'array can have either 3 or 1 channel (i.e 3rd dimension)')
    else:
        if len(array.shape) != 2:
            raise ValueError(
                'Array must be 3 (with channels) or 2 dimensional')

    return PIL.Image.fromarray(array)
