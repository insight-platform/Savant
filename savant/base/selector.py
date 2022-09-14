"""Base object selectors."""
from abc import abstractmethod
import numpy as np
from savant.base.pyfunc import BasePyFuncCallableImpl


class BaseSelector(BasePyFuncCallableImpl):
    """Selector interface."""

    @abstractmethod
    def __call__(self, bbox_tensor: np.ndarray) -> np.ndarray:
        """Filters objects.

        :param bbox_tensor: Bounding boxes for selection, represented as numpy array
            and contains ``(class_id, confidence, xc, yc, width, height, [angle])``
        :return: Resulting BBox tensor which contains only the selected boxes
        """
