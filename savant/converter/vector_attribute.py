"""Tensor to vector converters."""

from typing import List, Optional, Tuple

import numpy as np

from savant.base.converter import BaseAttributeModelOutputConverter
from savant.base.model import AttributeModel


class TensorToVectorConverter(BaseAttributeModelOutputConverter):
    """Tensor to vector converter.

    Eg ReID model.
    """

    def __call__(
        self,
        *output_layers: np.ndarray,
        model: AttributeModel,
        roi: Tuple[float, float, float, float],
    ) -> Optional[List[Tuple[str, List[float], float]]]:
        """Converts output array to Python list."""
        output_layer = output_layers[0]
        attr_config = model.output.attributes[0]

        return [(attr_config.name, output_layer.tolist(), 1.0)]


class TensorToItemConverter(BaseAttributeModelOutputConverter):
    """Tensor to item converter."""

    def __call__(
        self,
        *output_layers: np.ndarray,
        model: AttributeModel,
        roi: Tuple[float, float, float, float],
    ) -> Optional[List[Tuple[str, float, float]]]:
        """Converts output arrays to floats."""
        return [
            (attr.name, out.item(), 1.0)
            for out, attr in zip(output_layers, model.output.attributes)
        ]
