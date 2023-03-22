"""Person-face matching module."""
from typing import List
import numpy as np
from scipy.optimize import linear_sum_assignment


def __calc_intersection_areas(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Calculates intersection areas for two sets of boxes.

    :param boxes_a: Bbox array shape (n,4), coordinates order [x_min, y_min, x_max, y_max]
    :param boxes_b: Bbox array shape (m,4), coordinates order [x_min, y_min, x_max, y_max]
    :return: np.ndarray shape (n,m), rows for boxes a, cols for boxes b
    """

    # insert new axis to reshape column from boxes_a to two dimensions (n,1)
    # then broadcast column from b, result will be matrix(n,m)
    intersections_lefts = np.maximum(boxes_a[:, np.newaxis, 0], boxes_b[:, 0])
    intersections_rights = np.minimum(boxes_a[:, np.newaxis, 2], boxes_b[:, 2])
    intersections_widths = np.clip(
        intersections_rights - intersections_lefts, a_min=0, a_max=None
    )

    intersections_tops = np.maximum(boxes_a[:, np.newaxis, 1], boxes_b[:, 1])
    intersections_bottoms = np.minimum(boxes_a[:, np.newaxis, 3], boxes_b[:, 3])
    intersections_heights = np.clip(
        intersections_bottoms - intersections_tops, a_min=0, a_max=None
    )

    return intersections_widths * intersections_heights


def __iou(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Calculates IoU coefficients for two sets of boxes.

    :param boxes_a: Bbox array shape (n,4), coordinates order [x_min, y_min, x_max, y_max]
    :param boxes_b: Bbox array shape (m,4), coordinates order [x_min, y_min, x_max, y_max]
    :return: np.ndarray shape (n,m), rows for boxes a, cols for boxes b
    """

    inter_areas = __calc_intersection_areas(boxes_a, boxes_b)

    wh_a = boxes_a[:, 2:] - boxes_a[:, :2]
    wh_b = boxes_b[:, 2:] - boxes_b[:, :2]

    areas_a = np.prod(wh_a, axis=1)
    areas_b = np.prod(wh_b, axis=1)

    union_areas = areas_a[:, np.newaxis] + areas_b - inter_areas

    return inter_areas / union_areas


def match_person_faces(person_boxes: np.ndarray, face_boxes: np.ndarray) -> List[int]:
    """Matches persons and faces based on their bounding boxes' IOU coefficients.

    :param person_boxes: Bbox array shape (n,4), coordinates order [x_min, y_min, x_max, y_max]
    :param face_boxes: Bbox array shape (n,4), coordinates order [x_min, y_min, x_max, y_max]
    :return: indexes of persons that were successfully matched with a face.
    """
    iou_matrix = __iou(person_boxes, face_boxes)
    row_idxs, col_idxs = linear_sum_assignment(iou_matrix, maximize=True)

    person_with_face_idxs = [
        row_idx
        for row_idx, col_idx in zip(row_idxs, col_idxs)
        if iou_matrix[row_idx, col_idx] > 0
    ]
    return person_with_face_idxs
