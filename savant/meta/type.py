"""Metadata enums."""


class InformationType:
    """Sets the correspondence between the type of information and the index of
    the array where this information is stored."""

    OBJECT_SELECTION_TYPE = 0
    """Sets the field that stores informationabout the type of object selection."""

    OBJECT_HASH_KEY = 1
    """Sets the field that stores the unique hash key for an object."""


class ObjectSelectionType:
    """Types of object selection on frames."""

    REGULAR_BBOX = 1
    """The object is selected on a frame using a regular (aligned) bounding box."""
    ROTATED_BBOX = 2
    """The object is selected on a frame using a rotated bounding box."""
