from typing import Union

from savant_rs.primitives.geometry import BBox, RBBox


def copy_bbox(bbox: Union[BBox, RBBox]) -> Union[BBox, RBBox]:
    if isinstance(bbox, BBox):
        # BBox doesn't have method "copy".
        # Method "copy_py" doesn't work in this case:
        # modification of the copy affects the original.
        return BBox(bbox.xc, bbox.yc, bbox.width, bbox.height)

    return bbox.copy()
