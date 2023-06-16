"""Metadata constants."""
import numpy as np


UNTRACKED_OBJECT_ID = np.iinfo(np.uint64).max
"""This track id is assigned to objects that have not been tracked."""

DEFAULT_CONFIDENCE = 1.0
"""Default confidence value."""

DEFAULT_MODEL_NAME = 'auto'
"""Default model name."""

PRIMARY_OBJECT_LABEL = 'frame'
"""Primary object label (frame RoI object)."""

# TODO: replace separator on constant from savant-rs
PRIMARY_OBJECT_KEY = f'{DEFAULT_MODEL_NAME}.{PRIMARY_OBJECT_LABEL}'
"""Object key for primary object."""
