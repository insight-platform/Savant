"""Model raw output converter."""
from typing import Any, List, Optional, Tuple

import numpy as np

from savant.base.converter import BaseAttributeModelOutputConverter
from savant.base.model import AttributeModel


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
