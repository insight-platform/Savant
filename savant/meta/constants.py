"""Metadata constants."""
from savant_rs.utils.symbol_mapper import build_model_object_key

UNTRACKED_OBJECT_ID = 2**64 - 1
"""This track id is assigned to objects that have not been tracked."""

DEFAULT_CONFIDENCE = 1.0
"""Default confidence value."""

DEFAULT_MODEL_NAME = 'auto'
"""Default model name."""

PRIMARY_OBJECT_LABEL = 'frame'
"""Primary object label (frame RoI object)."""

PRIMARY_OBJECT_KEY = build_model_object_key(DEFAULT_MODEL_NAME, PRIMARY_OBJECT_LABEL)
"""Object key for primary object."""
