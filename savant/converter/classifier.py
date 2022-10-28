"""Tensor to label converter."""
from typing import List, Tuple, Optional
import numpy as np
from savant.base.converter import BaseAttributeModelOutputConverter
from savant.base.model import AttributeModel


class TensorToLabelConverter(BaseAttributeModelOutputConverter):
    """Raw classifier output to label converter."""

    def __init__(self, apply_softmax: bool = False, **kwargs):
        """
        :param softmax: Apply softmax function to output.
        """
        super().__init__(**kwargs)
        self.apply_softmax = apply_softmax

    def __call__(
        self,
        *output_layers: np.ndarray,
        model: AttributeModel,
        roi: Tuple[float, float, float, float]
    ) -> List[Tuple[str, str, Optional[float]]]:
        """Converts attribute (complex) model output layer values to
        ``(attr_name, label, confidence)`` tuples."""
        result = []
        for values, attr_config in zip(output_layers, model.output.attributes):
            multi_label = attr_config.multi_label
            # values are out of range (0, 1) - raw output? - apply softmax
            # (to get valid confidence)
            if self.apply_softmax or np.any(values > 1.0):
                values = softmax(values)
                multi_label = False

            if multi_label:
                idx_values = list(enumerate(values))
            else:
                idx = np.argmax(values)
                idx_values = [(idx, values[idx])]

            for idx, value in idx_values:
                label = idx
                if attr_config.labels:
                    label = attr_config.labels[idx]
                if attr_config.threshold is None or value > attr_config.threshold:
                    result.append((attr_config.name, str(label), float(value)))

        return result


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()
