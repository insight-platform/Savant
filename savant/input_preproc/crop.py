"""Model input metadata preprocessing: cropping variations."""
import pyds
from savant.base.input_preproc import BasePreprocessObjectMeta


#  TODO Implement crop
class CropTopPreprocessObjectMeta(BasePreprocessObjectMeta):
    """Make bbox height no greater that bbox width."""

    def __call__(self, bbox: pyds.NvBbox_Coords) -> pyds.NvBbox_Coords:
        bbox_height = bbox.width
        if bbox_height > bbox.height:
            bbox_height = bbox.height
        bbox.height = bbox_height
        return bbox
