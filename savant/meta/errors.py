"""Metadata exceptions."""


class IncorrectSelectionType(Exception):
    """Exception on errors when working with type of selection object on
    frame."""


class BaseMetaException(Exception):
    """Base exception when working with meta."""


class MetaValueError(BaseMetaException):
    """Exception on errors when working with meta and pass incorrect value."""


class UIDError(BaseMetaException):
    """Exception on errors when working with the unique object id."""


class MetaTypeError(BaseMetaException):
    """Exception on errors when working with incorrect meta type."""


class MetaPoolError(BaseMetaException):
    """Exception on errors when working with meta pool."""
