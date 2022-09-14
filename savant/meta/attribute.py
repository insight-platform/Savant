"""Attribute metadata."""
from typing import Any
from dataclasses import dataclass


@dataclass
class AttributeMeta:
    """Attribute meta information."""

    element_name: str
    """Element name that created the attribute."""

    name: str
    """Attribute name."""

    value: Any
    """Attribute value."""

    confidence: float
    """Attribute confidence."""
