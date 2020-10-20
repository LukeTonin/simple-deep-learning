"""This module contains utility functions for performning manipulations of images stored as arrays.
"""

from typing import List

import numpy as np

from ..bounding_box import format_bounding_box, calculate_iou


def overlay_arrays(array_shape: tuple,
                   input_arrays: np.ndarray,
                   input_labels: np.ndarray,
                   num_input_arrays_to_overlay: int,
                   max_array_value: int,
                   max_iou: float = 0.2,
                   ):
    """Generate an array by overlaying multiple smaller arrays onto a blank one.

    The smaller arrays are randomly selected from input_arrays which
    is an array of all the smaller arrays stacked along the axis 0.

    Parameters:
        array_shape: The shape of the output array.
        input_arrays: Each row corresponds to an input array that
            can be overlayed on the output array.
        input_labels: Same thing as input_arrays but contains the labels.
        num_input_arrays_to_overlay: The number of arrays to attempt to 
           add to the output array.
        max_array_value: The maximum allowed value for this array.
            Any number larger than this will be clipped.
            Clipping is necessary because the overlaying is done by summing arrays.
        max_iou: The maximum allowed IOU between two overlayed arrays.

    Returns:
        output_array: The output array of size array_shape.
        arrays_used: an array of shape (num_arrays_overlayed, input_array_shape)
        labels_overlayed: an array of shape (num_images_overlayed, input_label_shape)
        bounding_boxes_overlayed: an array of shape (num_images_overlayed, 4)
            The bounding boxes are absolute pixel values in the format xmin, ymin, xmax, ymax
    """

    output_array = np.zeros(array_shape)

    indices = np.random.randint(
        len(input_arrays), size=num_input_arrays_to_overlay)
    bounding_boxes = []
    bounding_boxes_as_tuple = []
    indices_overlayed = []
    for i in indices:
        bounding_box = overlay_at_random(
            array1=output_array, array2=input_arrays[i],
            max_array_value=max_array_value,
            bounding_boxes=bounding_boxes, max_iou=max_iou)

        if bounding_box is None:
            break

        indices_overlayed.append(i)

        bounding_boxes_as_tuple.append(
            format_bounding_box(bounding_box, output_type='tuple'))
        bounding_boxes.append(bounding_box)

    arrays_overlayed = input_arrays[indices_overlayed]
    labels_overlayed = input_labels[indices_overlayed]
    bounding_boxes_overlayed = np.stack(bounding_boxes_as_tuple)

    return output_array, arrays_overlayed, labels_overlayed, bounding_boxes_overlayed


def overlay_at_random(array1: np.ndarray, array2: np.ndarray,
                      max_array_value: int,
                      bounding_boxes: List[dict] = None,
                      max_iou: float = 0.2) -> np.ndarray:
    """Overlay an array over another.

    Overlays array2 over array1 while attempting to avoid locations specified by 
    a list of bounding_boxes. This function overlays inplace so array1 is not
    copied or returned.

    THe location of the array2 in array1 is determined at random.

    Parameters:
        array1: The base array (or canvas) on which to overlay array2.
        array2: The second array to overlay over array1.
        max_array_value: The maximum allowed value for this array.
            Any number larger than this will be clipped.
            Clipping is necessary because the overlaying is done by summing arrays.
        bounding_boxes: A list of bounding boxes in the format xyxy.
           The algorithm will not overlay with existing bounding boxes by more
           than an IOU of max_iou.
        max_iou: The maximum allowed IOU between the candidate location and the
            bounding_boxes.

    Returns:
        The bounding box of the added image if successfully overlayed. Otherwise None.
    """
    if not bounding_boxes:
        bounding_boxes = []

    height1, width1, *_ = array1.shape
    height2, width2, *_ = array2.shape

    max_x = width1 - width2
    max_y = height1 - height2

    is_valid = False
    # This number is arbitrary. There are better ways of doing this but this is fast enough.
    max_attempts = 1000
    attempt = 0
    while not is_valid:
        if attempt > max_attempts:
            return
        else:
            attempt += 1
        x = np.random.randint(max_x + 1)
        y = np.random.randint(max_y + 1)

        candidate_bounding_box = {
            'xmin': x,
            'ymin': y,
            'xmax': x + width2,
            'ymax': y + height2,
        }

        is_valid = True
        for bounding_box in bounding_boxes:
            if calculate_iou(bounding_box, candidate_bounding_box) > max_iou:
                is_valid = False
                break

    overlay_array(array1=array1, array2=array2, x=x, y=y, max_array_value=max_array_value)

    return candidate_bounding_box


def overlay_array(array1: np.ndarray, array2: np.ndarray, x: int, y: int, max_array_value: int = None) -> np.ndarray:
    """Overlay an array on another at a given position.

    Parameters:
        array1: The base array (or canvas) on which to overlay array2.
        array2: The second array to overlay over array1.
        max_array_value: The maximum allowed value for this array.
            Any number larger than this will be clipped.
            Clipping is necessary because the overlaying is done by summing arrays.
    
    Returns:
        array1: array1 with array2 overlayed at the position x, y.

    """

    height1, width1, *other1 = array1.shape
    height2, width2, *other2 = array2.shape

    if height2 > height1 or width2 > width1:
        raise ValueError('array2 must have a smaller shape than array1')

    if other1 != other2:
        raise ValueError('array1 and array2 must have same dimensions beyond dimension 2.')

    array1[y:y+height2, x:x+width2, ...] = array1[y:y + height2, x:x+width2, ...] + array2

    array1 = np.clip(array1, a_min=0, a_max=max_array_value, out=array1)

    return array1
