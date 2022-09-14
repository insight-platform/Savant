"""Parameter storage interface."""
from abc import ABC, abstractmethod
from typing import Any, Callable, Optional


class ParameterStorage(ABC):
    """Parameter storage interface."""

    @abstractmethod
    def __setitem__(self, name: str, value: Any):
        """Set value for a given name."""

    @abstractmethod
    def __getitem__(self, name: str) -> Any:
        """Get value for a given name.

        Name is expected to be registered first.
        """

    @abstractmethod
    def register_parameter(
        self, name: str, default_value: Optional[Any] = None
    ) -> None:
        """Registers static parameter."""

    @abstractmethod
    def register_dynamic_parameter(
        self,
        name: str,
        default_value: Optional[Any] = None,
        on_change: Optional[Callable] = None,
    ) -> None:
        """Register dynamic parameter."""
