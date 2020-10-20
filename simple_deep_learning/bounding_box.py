"""This module contains utility functions related to bounding boxes.
"""
from typing import Union

import numpy as np


def format_bounding_box(bounding_box: Union[tuple, dict, np.ndarray, list],
                        input_format: str = None,
                        output_format: str = 'xyxy',
                        output_type: str = 'dict') -> Union[dict, tuple]:
    """Format a bounding box object.

    This is a utility function for converting bounding boxes between different formats.
    There are two caracteristics for a bounding box:
    - format: Whether the bounding box is defined by its min and max values: xmin, ymin, xmax, ymax
        or by its minimum x and y and a width and height.
    - type: Whether or not the bounding box is an indexable (e.g tuple, array, list) or a dictionary.

    This function converts between all types.

    In the case of output_type == 'dict', the keys of the dictionary will be
    reordered and renamed to be either:
    - xmin. ymin, xmax, ymax for the format xyxy.
    - x, y, width, height for the format xywh.

    Parameters:
        bounding_box: The input bounding box, as a tuple, array, list or dictionary.
        input_format: The format of the input. Required if the input type is tuple.
            Otherwise the input format is inferred from the keys of the dictionary.
        output_format: Determines the output format of the bounding box.
            Must be 'xyxy' or 'xywh'. Defaults to 'xyxy'
        output_type: The output type of the bounding box.
            Must be 'dict' or 'tuple'. Defaults to 'dict'.

    Returns:
        return_value: A bounding boxes represented in the specified format and type.
    """
    if output_format == 'xyxy':
        if isinstance(bounding_box, dict):
            if all(key in bounding_box for key in ['xmin', 'ymin', 'xmax', 'ymax']):
                return_value = {
                    'xmin': bounding_box['xmin'],
                    'ymin': bounding_box['ymin'],
                    'xmax': bounding_box['xmax'],
                    'ymax': bounding_box['ymax']
                }
            elif all(key in bounding_box for key in ['xmin', 'ymin', 'width', 'height']):
                return_value = {
                    'xmin': bounding_box['xmin'],
                    'ymin': bounding_box['ymin'],
                    'xmax': bounding_box['xmin'] + bounding_box['width'],
                    'ymax': bounding_box['ymin'] + bounding_box['height']
                }
            elif all(key in bounding_box for key in ['x', 'y', 'width', 'height']):
                return_value = {
                    'xmin': bounding_box['x'],
                    'ymin': bounding_box['y'],
                    'xmax': bounding_box['x'] + bounding_box['width'],
                    'ymax': bounding_box['y'] + bounding_box['height']
                }
            else:
                raise ValueError(
                    f'Incorrect format for bounding_box dictionary. Received: {bounding_box}')
        else:
            if input_format == 'xyxy':
                return_value = {
                    'xmin': bounding_box[0],
                    'ymin': bounding_box[1],
                    'xmax': bounding_box[2],
                    'ymax': bounding_box[3]
                }
            elif input_format == 'xywh':
                return_value = {
                    'xmin': bounding_box[0],
                    'ymin': bounding_box[1],
                    'xmax': bounding_box[0] + bounding_box[2],
                    'ymax': bounding_box[1] + bounding_box[3]
                }
            else:
                raise ValueError(
                    'If bounding_box is not a dictionary, input_format must be specified: "xyxy" or "xywh"')

    elif output_format == 'xywh':
        if isinstance(bounding_box, dict):
            if all(key in bounding_box for key in ['xmin', 'ymin', 'width', 'height']):
                return_value = {
                    'x': bounding_box['xmin'],
                    'y': bounding_box['ymin'],
                    'width': bounding_box['width'],
                    'height': bounding_box['height']
                }
            elif all(key in bounding_box for key in ['xmin', 'ymin', 'xmax', 'ymax']):
                return_value = {
                    'x': bounding_box['xmin'],
                    'y': bounding_box['ymin'],
                    'width': bounding_box['xmax'] - bounding_box['xmin'],
                    'height': bounding_box['ymax'] - bounding_box['ymin']
                }
            elif all(key in bounding_box for key in ['x', 'y', 'width', 'height']):
                return_value = {
                    'x': bounding_box['x'],
                    'y': bounding_box['y'],
                    'width': bounding_box['width'],
                    'height': bounding_box['height']
                }
            else:
                raise ValueError(
                    f'Incorrect format for bounding_box dictionary. Received: {bounding_box}')
        else:
            if input_format == 'xyxy':
                return_value = {
                    'x': bounding_box[0],
                    'y': bounding_box[1],
                    'width': bounding_box[2] - bounding_box[0],
                    'height': bounding_box[3] - bounding_box[1]
                }
            elif input_format == 'xywh':
                return_value = {
                    'x': bounding_box[0],
                    'y': bounding_box[1],
                    'width': bounding_box[2],
                    'height': bounding_box[3]
                }
            else:
                raise ValueError(
                    'If bounding_box is not a dictionary, input_format must be specified: "xyxy" or "xywh"')
    else:
        raise ValueError(
            f'output_format must be either "xyxy" or "xywh". Received {output_format}')

    if output_type == 'tuple':
        return tuple(return_value.values())
    elif output_type == 'dict':
        return return_value
    else:
        raise ValueError(
            f'output_type must be either "dict" or "tuple". Received {output_type}')


def calculate_iou(bounding_box1: dict, bounding_box2: dict) -> float:
    """Calculate the intersection over union of two bounding boxes.

    Both bounding boxes must be in xyxy format and of type dict.
    See format_bounding_box function for more details.

    Returns:
        IOU: number between 0 and 1.
    """

    A1 = ((bounding_box1['xmax'] - bounding_box1['xmin'])
          * (bounding_box1['ymax'] - bounding_box1['ymin']))
    A2 = ((bounding_box2['xmax'] - bounding_box2['xmin'])
          * (bounding_box2['ymax'] - bounding_box2['ymin']))

    xmin = max(bounding_box1['xmin'], bounding_box2['xmin'])
    ymin = max(bounding_box1['ymin'], bounding_box2['ymin'])
    xmax = min(bounding_box1['xmax'], bounding_box2['xmax'])
    ymax = min(bounding_box1['ymax'], bounding_box2['ymax'])

    if ymin >= ymax or xmin >= xmax:
        return 0

    return ((xmax-xmin) * (ymax - ymin)) / (A1 + A2)
