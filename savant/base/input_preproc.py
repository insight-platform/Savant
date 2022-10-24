"""Base model input preprocessors."""
from abc import abstractmethod
from typing import Optional
import pyds
from savant.base.pyfunc import BasePyFuncCallableImpl


class BasePreprocessObjectMeta(BasePyFuncCallableImpl):
    """Object meta preprocessing interface."""

    @abstractmethod
    def __call__(
        self,
        bbox: pyds.NvBbox_Coords,
        *,
        parent_bbox: Optional[pyds.NvBbox_Coords] = None,
        **kwargs
    ) -> pyds.NvBbox_Coords:
        """Transforms object meta.

        :param bbox: original bbox
        :param parent_bbox: parent object bbox, eg frame
        :return: changed bbox
        """
