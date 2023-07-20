"""Analytics module."""
from typing import Dict, Tuple

from savant.deepstream.utils import nvds_obj_meta_iterator
from savant.gstreamer import Gst
from savant.deepstream.meta.frame import NvDsFrameMeta
from savant.deepstream.pyfunc import NvDsPyFuncPlugin
from savant.meta.constants import PRIMARY_OBJECT_LABEL  # , PRIMARY_OBJECT_KEY
from savant.parameter_storage import param_storage


class IOU(NvDsPyFuncPlugin):
    """Age and gender smoothing pyfunc.
    On each frame

    :key history_length: int, Length of the vector of historical values
    """

    def __init__(self, ground_truth: str, element_name: str, **kwargs):
        super().__init__(**kwargs)
        self.element_name = element_name
        self.ground_truth = ground_truth

    def process_frame(self, buffer: Gst.Buffer, frame_meta: NvDsFrameMeta):
        """Process frame metadata.

        :param buffer: Gstreamer buffer with this frame's data.
        :param frame_meta: This frame's metadata.
        """
        ground_truth_objects = []
        detected_objects = []
        for obj in frame_meta.objects:
            if obj.element_name == self.element_name:
                detected_objects.append(obj)
            elif obj.element_name == self.ground_truth:
                ground_truth_objects.append(obj)
        if len(ground_truth_objects) == 1 and len(detected_objects) == 1:
            frame_meta.tags["iou_metric"] = ground_truth_objects[0].bbox.iou(
                detected_objects[0].bbox
            )
        else:
            frame_meta.tags["iou_metric"] = 0
