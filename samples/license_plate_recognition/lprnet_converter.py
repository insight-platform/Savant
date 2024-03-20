"""LPRNet model output converter."""
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np

from savant.base.converter import BaseAttributeModelOutputConverter
from savant.base.model import AttributeModel


class LPRNetOutputConverter(BaseAttributeModelOutputConverter):
    """OCRNet output converter."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        with open(Path(__file__).parent.resolve() / 'dict_us.txt', 'r') as dict_fp:
            self._char_list = dict_fp.read().split('\n')

    def __call__(
        self,
        *output_layers: np.ndarray,
        model: AttributeModel,
        roi: Tuple[float, float, float, float],
    ) -> List[Tuple[str, Any, Optional[float]]]:
        """Converts output tensors to attribute values.
        TODO: numba?

        :param output_layers: Model output layer tensors
        :param model: Attribute model
        :param roi: ``[top, left, width, height]`` of the rectangle
            on which the model infers
        :return: list of attributes values with confidences
            ``(attr_name, value, confidence)``
        """
        output_char_idx = output_layers[0]
        output_prob = output_layers[1]
        output_len = len(output_char_idx)
        char_list_len = len(self._char_list)

        text = ''
        prob = 1.0
        prev_char_idx = output_char_idx[0]
        if 0 <= prev_char_idx <= char_list_len:
            text += self._char_list[prev_char_idx]
            prob *= output_prob[0]

        for i in range(1, output_len):
            curr_char_idx = output_char_idx[i]
            if not (0 <= curr_char_idx <= char_list_len):
                continue

            if curr_char_idx != prev_char_idx:
                if curr_char_idx != char_list_len:
                    text += self._char_list[curr_char_idx]
                    prob *= output_prob[i]
                prev_char_idx = curr_char_idx

        if len(text) <= 3 or prob < model.output.attributes[0].threshold:
            return []

        return [(model.output.attributes[0].name, text, prob)]
