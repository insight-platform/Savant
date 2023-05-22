"""Metadata constants."""
import numpy as np


UNTRACKED_OBJECT_ID = np.iinfo(np.uint64).max
"""This track id is assigned to objects that have not been tracked."""

DEFAULT_CONFIDENCE = 1.0
"""Default confidence value."""

PRIMARY_OBJECT_LABEL = 'frame'
"""Primary object label (frame RoI object)."""
