"""Deepstream bounding boxes."""
import pyds
from pysavantboost import get_rbbox

from savant.meta.errors import MetaValueError
from savant.meta.bbox import BBox, RBBox


class NvDsBBox(BBox):
    """Deepstream bounding box wrapper.

    :param nvds_object_meta: Deepstream pyds object meta.
    """

    def __init__(self, nvds_object_meta: pyds.NvDsObjectMeta):
        if (
            nvds_object_meta.tracker_bbox_info.org_bbox_coords.width > 0
            and nvds_object_meta.tracker_bbox_info.org_bbox_coords.height > 0
        ):
            self._nv_ds_bbox = nvds_object_meta.tracker_bbox_info.org_bbox_coords
        elif (
            nvds_object_meta.detector_bbox_info.org_bbox_coords.width > 0
            and nvds_object_meta.detector_bbox_info.org_bbox_coords.height > 0
        ):
            self._nv_ds_bbox = nvds_object_meta.detector_bbox_info.org_bbox_coords
        else:
            raise MetaValueError(
                'No valid bounding box found in Deepstream meta information.'
            )
        self._nv_ds_rect_meta = nvds_object_meta.rect_params

    @property
    def x_center(self) -> float:
        """X coordinate of bounding box center point."""
        return self._nv_ds_bbox.left + 0.5 * self.width

    @x_center.setter
    def x_center(self, value: float):
        self._nv_ds_bbox.left = value - 0.5 * self.width
        self._nv_ds_rect_meta.left = self._nv_ds_bbox.left

    @property
    def y_center(self) -> float:
        """Y coordinate of bounding box center point."""
        return self._nv_ds_bbox.top + 0.5 * self.height

    @y_center.setter
    def y_center(self, value: float):
        self._nv_ds_bbox.top = value - 0.5 * self.height
        self._nv_ds_rect_meta.top = self._nv_ds_bbox.top

    @property
    def width(self) -> float:
        """Width of bounding box."""
        return self._nv_ds_bbox.width

    @width.setter
    def width(self, value: float):
        self._nv_ds_bbox.width = value
        self._nv_ds_rect_meta.width = value

    @property
    def height(self) -> float:
        """Height of bounding box."""
        return self._nv_ds_bbox.height

    @height.setter
    def height(self, value: float):
        self._nv_ds_bbox.height = value
        self._nv_ds_rect_meta.height = value

    @property
    def top(self) -> float:
        """Y coordinate of the upper left corner."""
        return self._nv_ds_bbox.top

    @top.setter
    def top(self, value: float):
        self._nv_ds_bbox.top = value
        self._nv_ds_rect_meta.top = value

    @property
    def left(self) -> float:
        """X coordinate of the upper left corner."""
        return self._nv_ds_bbox.left

    @left.setter
    def left(self, value: float):
        self._nv_ds_bbox.left = value
        self._nv_ds_rect_meta.left = value


class NvDsRBBox(RBBox):
    """Deepstream rotated bounding box wrapper.

    :param nvds_object_meta: Deepstream pyds object meta.
    """

    def __init__(self, nvds_object_meta: pyds.NvDsObjectMeta):
        self._rbbox = get_rbbox(nvds_object_meta)
        if self._rbbox is None:
            raise MetaValueError(
                'No rotated bounding box found in Deepstream meta information.'
            )

    @property
    def x_center(self) -> float:
        """X coordinate of bounding box center point."""
        return self._rbbox.x_center

    @x_center.setter
    def x_center(self, value: float):
        self._rbbox.x_center = value

    @property
    def y_center(self) -> float:
        """Y coordinate of bounding box center point."""
        return self._rbbox.y_center

    @y_center.setter
    def y_center(self, value: float):
        self._rbbox.y_center = value

    @property
    def width(self) -> float:
        """Width of bounding box."""
        return self._rbbox.width

    @width.setter
    def width(self, value: int):
        self._rbbox.width = value

    @property
    def height(self) -> float:
        """Height of bounding box."""
        return self._rbbox.height

    @height.setter
    def height(self, value: float):
        self._rbbox.height = value

    @property
    def angle(self) -> float:
        """Angle of bbox rotation."""
        return self._rbbox.angle

    @angle.setter
    def angle(self, value: float):
        self._rbbox.angle = value
