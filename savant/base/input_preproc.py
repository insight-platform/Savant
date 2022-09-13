"""Base model input preprocessors."""
from abc import abstractmethod
import pyds
from savant.base.pyfunc import BasePyFuncCallableImpl


class BasePreprocessObjectMeta(BasePyFuncCallableImpl):
    """Object meta preprocessing interface."""

    @abstractmethod
    def __call__(self, bbox: pyds.NvBbox_Coords) -> pyds.NvBbox_Coords:
        """Transforms object meta.

        :param bbox: original bbox
        :return: changed bbox
        """
