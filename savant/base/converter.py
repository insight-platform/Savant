"""Base model output converters."""
from abc import abstractmethod
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import cupy as cp

from savant.base.model import AttributeModel, ComplexModel, ObjectModel
from savant.base.pyfunc import BasePyFuncCallableImpl


class BaseOutputConverter(BasePyFuncCallableImpl):
    """Base model output converter."""

    gpu: bool = False
    """Set to True to get the ``output_layers'' tensors in the converter call 
    on the GPU (cupy.ndarray), not the CPU (numpy.ndarray).
    """

    @abstractmethod
    def __call__(
        self,
        *output_layers: Union[np.ndarray, cp.ndarray],
        model: ObjectModel,
        roi: Tuple[float, float, float, float],
    ) -> Any:
        """Converts raw model output tensors to a model specific representation."""


class BaseObjectModelOutputConverter(BaseOutputConverter):
    """Base object model output converter."""

    @abstractmethod
    def __call__(
        self,
        *output_layers: Union[np.ndarray, cp.ndarray],
        model: ObjectModel,
        roi: Tuple[float, float, float, float],
    ) -> np.ndarray:
        """Converts raw model output tensors to a numpy array that represents a
        list of detected bboxes in the format ``(class_id, confidence, xc, yc,
        width, height, [angle])`` in absolute coordinates computed with ``ROI``
        information.

        :param output_layers: Model output layer tensors
        :param model: Object model, required parameters: input tensor shape,
            maintain_aspect_ratio flag
        :param roi: ``[top, left, width, height]`` of the rectangle
            on which the model infers
        :return: BBox tensor ``(class_id, confidence, xc, yc, width, height, [angle])``
            offset by roi upper left and scaled by roi width and height
        """


class BaseAttributeModelOutputConverter(BaseOutputConverter):
    """Base attribute model output converter."""

    @abstractmethod
    def __call__(
        self,
        *output_layers: Union[np.ndarray, cp.ndarray],
        model: AttributeModel,
        roi: Tuple[float, float, float, float],
    ) -> List[Tuple[str, Any, Optional[float]]]:
        """Converts raw model output tensors to a list of values in several
        formats:

        * classification
            * string label
            * int or float value
        * ReID
            * extracted descriptor as a vector of values

        and so on, depending on the task.

        :param output_layers: Model output layer tensors
        :param model: Attribute model
        :param roi: ``[top, left, width, height]`` of the rectangle
            on which the model infers
        :return: list of attributes values with confidences
            ``(attr_name, value, confidence)``
        """


class BaseComplexModelOutputConverter(BaseOutputConverter):
    """Base complex model output converter."""

    @abstractmethod
    def __call__(
        self,
        *output_layers: Union[np.ndarray, cp.ndarray],
        model: ComplexModel,
        roi: Tuple[float, float, float, float],
    ) -> Tuple[np.ndarray, List[List[Tuple[str, Any, float]]]]:
        """Converts raw model output tensors to Savant format.

        :param output_layers: Model output layer tensors
        :param model: Complex model, required parameters: input tensor shape,
            maintain_aspect_ratio flag
        :param roi: ``[top, left, width, height]`` of the rectangle
            on which the model infers
        :return: a combination of :py:class:`.BaseObjectModelOutputConverter` and
            :py:class:`.BaseAttributeModelOutputConverter` outputs:

            * BBox tensor ``(class_id, confidence, xc, yc, width, height, [angle])``
              offset by roi upper left and scaled by roi width and height,
            * list of attributes values with confidences
              ``(attr_name, value, confidence)``
        """
