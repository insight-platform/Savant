
from typing import Any, List, Optional, Tuple

import numpy as np
import cv2
from savant.base.converter import BaseAttributeModelOutputConverter
from savant.base.model import AttributeModel



class AnimeganConverter(BaseAttributeModelOutputConverter):
    def __call__(
        self,
        *output_layers: np.ndarray,
        model: AttributeModel,
        roi: Tuple[float, float, float, float]
    ) -> List[Tuple[str, Any, Optional[float]]]:

        img = output_layers[0]
        img = (img + 1) / 2 * 255
        img = np.rint(img)
        img = np.clip(img, 0, 255).astype(np.uint8)
        # chw to hwc
        img = np.transpose(img, (1, 2, 0))

        img = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)
        # self.logger.info('host shape %s, dtype %s, min/max %s / %s', img.shape, img.dtype, np.amin(img), np.amax(img) )
        return [
            (model.output.attributes[0].name, img, None),
        ]
