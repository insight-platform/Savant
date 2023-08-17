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
        roi: Tuple[float, float, float, float]
    ) -> List[Tuple[List[float], Optional[float]]]:
        """Converts output array to Python list."""
        return [(output_layers[0].tolist(), None)]


class TensorToItemConverter(BaseAttributeModelOutputConverter):
    """Tensor to item converter."""

    def __call__(
        self,
        *output_layers: np.ndarray,
        model: AttributeModel,
        roi: Tuple[float, float, float, float]
    ) -> List[Tuple[float, Optional[float]]]:
        """Converts output arrays to floats."""
        return [(out.item(), None) for out in output_layers]
