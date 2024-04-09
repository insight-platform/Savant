from typing import Any, List, Optional, Tuple

import cv2
import numpy as np

from savant.base.converter import BaseAttributeModelOutputConverter
from savant.base.model import AttributeModel


class AnimeganConverter(BaseAttributeModelOutputConverter):
    def __call__(
        self,
        *output_layers: np.ndarray,
        model: AttributeModel,
        roi: Tuple[float, float, float, float],
    ) -> List[Tuple[str, Any, Optional[float]]]:

        img = output_layers[0]
        # from [-1, 1] to [0, 255]
        img = (img + 1) * 127.5
        img = np.rint(img).clip(0, 255).astype(np.uint8)
        # chw to hwc
        img = np.transpose(img, (1, 2, 0))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)
        return [
            (model.output.attributes[0].name, img, None),
        ]
