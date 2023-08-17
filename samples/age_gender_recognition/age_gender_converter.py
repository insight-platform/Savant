"""Age gender converter module."""
from typing import Any, List, Optional, Tuple

import numpy as np

from savant.base.converter import BaseAttributeModelOutputConverter
from savant.base.model import AttributeModel
from savant.parameter_storage import param_storage

GENDER_MAPPING = ['male', 'female']

age_min = param_storage()['age_min']
age_max = param_storage()['age_max']
age_range = np.arange(age_min, age_max)


class AgeGenderConverter(BaseAttributeModelOutputConverter):
    def __call__(
        self,
        *output_layers: np.ndarray,
        model: AttributeModel,
        roi: Tuple[float, float, float, float]
    ) -> List[Tuple[str, Any, Optional[float]]]:
        """Converts age gender model output vector to age and gender attribute."""
        age = np.sum(np.multiply(output_layers[0], age_range)).item()
        gen = int(np.argmax(output_layers[1]).item())
        return [
            (model.output.attributes[0].name, age, None),
            (
                model.output.attributes[1].name,
                GENDER_MAPPING[gen],
                float(output_layers[1][gen].item()),
            ),
        ]
