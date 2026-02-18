"""Python resolver for module.yml — derives ROI input object from shared constants."""

import os

from samples.meta_merge.src.meta_merge.roi_constants import (
    LEFT_ROI_LABEL,
    RIGHT_ROI_LABEL,
    ROI_LEFT,
    ROI_NAMESPACE,
)


def roi_input_object() -> str:
    """Return the fully qualified ROI object name for nvinfer input."""
    roi = os.environ.get('MODULE_ROI', 'left').lower()
    label = LEFT_ROI_LABEL if roi == ROI_LEFT else RIGHT_ROI_LABEL
    return f'{ROI_NAMESPACE}.{label}'
