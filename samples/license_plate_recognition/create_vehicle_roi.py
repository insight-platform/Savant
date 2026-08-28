"""Adds a proxy "vehicle" ROI object to every detected vehicle.

The detector assigns its own label to each COCO vehicle type (car, motorcycle,
bus, truck), so that downstream units can distinguish between them. An nvinfer
unit, however, can operate on a single parent object class only. To run the LPD
model on every vehicle type, a proxy child object with a common label is added
to each detected vehicle; the proxy occupies the same area as its parent
(trimmed to the frame) and is used as the input object for the LPD model.
"""

from savant_rs.primitives.geometry import BBox

from savant.deepstream.meta.frame import NvDsFrameMeta
from savant.deepstream.pyfunc import NvDsPyFuncPlugin
from savant.gstreamer import Gst  # noqa: F401
from savant.meta.object import ObjectMeta
from savant.parameter_storage import param_storage


class CreateVehicleROI(NvDsPyFuncPlugin):
    """PyFunc that creates a proxy ROI for every detected vehicle."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.vehicle_labels = set(param_storage()['vehicle_labels'])
        self.roi_element_name = param_storage()['vehicle_roi']['element_name']
        self.roi_label = param_storage()['vehicle_roi']['label']

    def process_frame(self, buffer: Gst.Buffer, frame_meta: NvDsFrameMeta):
        """Callback on each frame in a Deepstream pipeline batch."""

        # the frame objects are collected before adding new ones
        # to avoid modifying the metadata while iterating over it
        vehicles = [
            obj_meta
            for obj_meta in frame_meta.objects
            if obj_meta.label in self.vehicle_labels
        ]

        frame_width = frame_meta.video_frame.width
        frame_height = frame_meta.video_frame.height

        for vehicle in vehicles:
            # the tracker may return bboxes that stick out of the frame
            # (see enableBboxUnClipping in the tracker config), while a proxy
            # is only valid as a model input if it fits in the viewport
            left, top, right, bottom = vehicle.bbox.as_ltrb()
            left, top = max(left, 0.0), max(top, 0.0)
            right, bottom = min(right, frame_width), min(bottom, frame_height)
            if right <= left or bottom <= top:
                continue

            roi_meta = ObjectMeta(
                element_name=self.roi_element_name,
                label=self.roi_label,
                bbox=BBox.ltrb(left, top, right, bottom),
                confidence=vehicle.confidence,
            )
            # the parent is assigned after the object is added to the frame:
            # if it is set beforehand, add_obj_meta() passes it to the DeepStream
            # object meta, whose parent setter reads the object uid, which
            # assigns one, and the uid cannot be assigned twice (UIDError)
            frame_meta.add_obj_meta(roi_meta)
            roi_meta.parent = vehicle
