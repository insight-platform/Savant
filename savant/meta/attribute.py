"""Attribute metadata."""
from dataclasses import dataclass
from typing import Any


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
