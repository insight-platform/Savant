"""Model raw output converters. They provide different options for processing output tensors: on the GPU or the host.
See `savant.base.converter.TensorFormat`"""

from typing import Any, List, Optional, Tuple

import cupy as cp
import numpy as np

from savant.base.converter import BaseAttributeModelOutputConverter, TensorFormat
from savant.base.model import AttributeModel


class ModelCudaRawOutputConverter(BaseAttributeModelOutputConverter):
    """Model raw output converter."""

    tensor_format: TensorFormat = TensorFormat.CuPy

    def __call__(
        self,
        *output_layers: cp.ndarray,
        model: AttributeModel,
        roi: Tuple[float, float, float, float],
    ) -> List[Tuple[str, Any, Optional[float]]]:
        """Returns raw model output tensors as attributes.

        :param output_layers: Model output layer tensors
        :param model: Attribute model
        :param roi: ``[top, left, width, height]`` of the rectangle
            on which the model infers
        :return: list of attributes values with confidences
            ``(attr_name, value, confidence)``
        """
        return [
            (model.output.attributes[i].name, output, 1.0)
            for i, output in enumerate(output_layers)
        ]


class ModelRawOutputConverter(BaseAttributeModelOutputConverter):
    """Model raw output converter."""

    def __call__(
        self,
        *output_layers: np.ndarray,
        model: AttributeModel,
        roi: Tuple[float, float, float, float],
    ) -> List[Tuple[str, Any, Optional[float]]]:
        """Returns raw model output tensors as attributes.

        :param output_layers: Model output layer tensors
        :param model: Attribute model
        :param roi: ``[top, left, width, height]`` of the rectangle
            on which the model infers
        :return: list of attributes values with confidences
            ``(attr_name, value, confidence)``
        """
        return [
            (model.output.attributes[i].name, output, 1.0)
            for i, output in enumerate(output_layers)
        ]
