"""Analytics module."""

from savant.deepstream.meta.frame import NvDsFrameMeta
from savant.deepstream.pyfunc import NvDsPyFuncPlugin
from savant.gstreamer import Gst


class IOU(NvDsPyFuncPlugin):
    """IOU metric for object detection."""

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
            frame_meta.set_tag(
                'iou_metric',
                ground_truth_objects[0].bbox.iou(detected_objects[0].bbox),
            )
        else:
            frame_meta.set_tag('iou_metric', 0)
