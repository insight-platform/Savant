"""OCRNet model output converter.
TODO: Move to model_utils.ocrnet_converter
"""

from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np

from savant.base.converter import BaseAttributeModelOutputConverter
from savant.base.model import AttributeModel


class OCRNetVITOutputConverter(BaseAttributeModelOutputConverter):
    """OCRNet output converter."""

    def __init__(self, confidence_threshold: float = 0.01, **kwargs):
        self.confidence_threshold = confidence_threshold
        super().__init__(**kwargs)
        self._character_list: Optional[List[str]] = None

    def _get_character_list(self, model_path: str):
        if self._character_list is None:
            with open(Path(model_path) / 'character_list', 'r') as f:
                self._character_list = ['[GO]', '[s]'] + f.read().split('\n')
        return self._character_list

    def __call__(
        self,
        *output_layers: np.ndarray,
        model: AttributeModel,
        roi: Tuple[float, float, float, float],
    ) -> Optional[List[Tuple[str, Any, float]]]:
        """Converts output tensors to attribute values.

        :param output_layers: Model output layer tensors
        :param model: Attribute model
        :param roi: ``[top, left, width, height]`` of the rectangle
            on which the model infers
        :return: list of attributes values with confidences
            ``(attr_name, value, confidence)``
        """
        output_id = output_layers[0]
        output_prob = output_layers[1]
        output_len = len(output_id)

        character_list = self._get_character_list(model.local_path)

        text = ''
        prob = 1.0
        probs = []

        for i in range(0, output_len):
            char = character_list[output_id[i]]
            if char != '[s]':
                text += char
                prob *= output_prob[i]
                probs.append(float(output_prob[i]))
            else:
                break

        if prob >= self.confidence_threshold:
            return [
                (model.output.attributes[0].name, text.upper(), prob),
                (model.output.attributes[0].name + '_probs', probs, 1.0),
            ]

        return None
