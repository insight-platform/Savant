"""Model input metadata preprocessing: cropping variations."""
from savant.base.input_preproc import BasePreprocessObjectMeta
from savant.meta.bbox import BBox
from savant.meta.object import ObjectMeta


#  TODO Implement crop
class CropTopPreprocessObjectMeta(BasePreprocessObjectMeta):
    """Make bbox height no greater that bbox width."""

    def __call__(self, object_meta: ObjectMeta) -> BBox:
        bbox = object_meta.bbox
        bbox_height = bbox.width
        if bbox_height > bbox.height:
            bbox_height = bbox.height
        bbox.y_center = bbox.top + bbox_height / 2
        bbox.height = bbox_height
        return bbox
