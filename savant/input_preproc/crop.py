"""Model input metadata preprocessing: cropping variations."""
from savant_rs.primitives.geometry import BBox

from savant.base.input_preproc import BasePreprocessObjectMeta
from savant.meta.object import ObjectMeta


class CropTopPreprocessObjectMeta(BasePreprocessObjectMeta):
    """Make bbox height no lesser that bbox width."""

    def __call__(self, object_meta: ObjectMeta) -> BBox:
        bbox = object_meta.bbox
        max_dim = max(bbox.width, bbox.height)
        return BBox(
            bbox.xc,
            bbox.top + max_dim / 2,
            bbox.width,
            max_dim,
        )
