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
            metric = ground_truth_objects[0].bbox.iou(detected_objects[0].bbox)
            frame_meta.set_tag(
                'iou_metric',
                metric,
            )
            self.logger.info(f'IOU metric: {metric}')
        else:
            frame_meta.set_tag('iou_metric', 0)
            self.logger.info(
                'GT count: %d, detected count: %d, IOU metric: 0',
                len(ground_truth_objects),
                len(detected_objects),
            )
