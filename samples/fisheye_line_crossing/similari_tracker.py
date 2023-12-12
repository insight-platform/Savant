"""Similari SORT tracker module."""
import math

from savant_rs.primitives.geometry import RBBox
from similari import PositionalMetricType, Sort, Universal2DBox

from savant.deepstream.meta.frame import NvDsFrameMeta
from savant.deepstream.pyfunc import NvDsPyFuncPlugin
from savant.gstreamer import Gst
from savant.meta.object import ObjectMeta
from savant.parameter_storage import param_storage

OBJ_LABEL = param_storage()['detected_object_label']


class Tracker(NvDsPyFuncPlugin):
    """."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        metric = PositionalMetricType.iou(threshold=self.iou_threshold)
        self.tracker = Sort(
            shards=4,
            bbox_history=2 * self.max_age,
            max_idle_epochs=self.max_age,
            method=metric,
        )

    def process_frame(self, buffer: Gst.Buffer, frame_meta: NvDsFrameMeta):
        """Process frame metadata.

        :param buffer: Gstreamer buffer with this frame's data.
        :param frame_meta: This frame's metadata.
        """
        objects = []
        tracker_inputs = []
        obj_idx = 0
        for obj in frame_meta.objects:
            if obj.label == OBJ_LABEL:
                xc, yc, width, height = obj.bbox.as_xcycwh()
                tracker_inputs.append(
                    (
                        Universal2DBox.new_with_confidence(
                            xc=xc,
                            yc=yc,
                            angle=obj.bbox.angle * math.pi / 180,
                            aspect=width / height,
                            height=height,
                            confidence=obj.confidence,
                        ),
                        obj_idx,
                    )
                )
                objects.append(obj)
                obj_idx += 1

        tracks = self.tracker.predict(tracker_inputs)

        for track in tracks:
            obj = objects[track.custom_object_id]

            if track.length < self.min_hits:
                frame_meta.remove_obj_meta(obj)
            else:
                track_bbox = track.predicted_bbox
                obj.track_id = track.id
                obj_bbox = obj.bbox
                obj_bbox.xc = track_bbox.xc
                obj_bbox.yc = track_bbox.yc
                obj_bbox.width = track_bbox.height * track_bbox.aspect
                obj_bbox.height = track_bbox.height
                if track_bbox.angle:
                    obj_bbox.angle = track_bbox.angle * 180 / math.pi
                else:
                    obj_bbox.angle = 0

        if self.add_idle:
            for track in self.tracker.idle_tracks():
                if track.length >= self.min_hits:
                    track_bbox = track.predicted_bbox
                    bbox = RBBox(
                        track_bbox.xc,
                        track_bbox.yc,
                        track_bbox.height * track_bbox.aspect,
                        track_bbox.height,
                        track_bbox.angle * 180 / math.pi,
                    )
                    frame_meta.add_obj_meta(
                        ObjectMeta(
                            'tracker',
                            OBJ_LABEL,
                            bbox,
                            track_bbox.confidence,
                            track.id,
                        )
                    )

        self.tracker.clear_wasted()
