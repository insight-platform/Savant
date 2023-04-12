from collections import defaultdict
import yaml
import numpy as np
from savant.gstreamer import Gst
from savant.deepstream.meta.frame import NvDsFrameMeta
from savant.deepstream.pyfunc import NvDsPyFuncPlugin
from samples.line_crossing.utils import TwoLinesCrossingTracker, Point, Direction

class LineCrossing(NvDsPyFuncPlugin):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        with open("/opt/app/samples/line_crossing/line_config.yml", "r", encoding='utf8') as stream:
            self.line_config = yaml.safe_load(stream)

        self.lc_trackers = {}
        self.entry_count = defaultdict(int)
        self.exit_count = defaultdict(int)
        self.cross_events = defaultdict(lambda: defaultdict(list))

    def on_source_eos(self, source_id: str):
        """On source EOS event callback."""
        if source_id in self.lc_trackers:
            del self.lc_trackers[source_id]
            del self.cross_events[source_id]
            del self.entry_count[source_id]
            del self.exit_count[source_id]

    def process_frame(self, buffer: Gst.Buffer, frame_meta: NvDsFrameMeta):
        """Process frame metadata.

        :param buffer: Gstreamer buffer with this frame's data.
        :param frame_meta: This frame's metadata.
        """
        primary_meta_object = None
        for obj_meta in frame_meta.objects:
            if obj_meta.is_primary:
                primary_meta_object = obj_meta

        if primary_meta_object and frame_meta.source_id in self.line_config:
            line_from = self.line_config[frame_meta.source_id]['from']
            line_to = self.line_config[frame_meta.source_id]['to']

            if frame_meta.source_id not in self.lc_trackers:
                self.lc_trackers[frame_meta.source_id] = TwoLinesCrossingTracker(
                    (Point(*line_from[:2]), Point(*line_from[2:])),
                    (Point(*line_to[:2]), Point(*line_to[2:]))
                )
            lc_tracker = self.lc_trackers[frame_meta.source_id]

            for obj_meta in frame_meta.objects:
                if obj_meta.label == 'person':
                    lc_tracker.add_track_point(
                        obj_meta.track_id,
                        # center point
                        Point(obj_meta.bbox.left + obj_meta.bbox.width / 2, obj_meta.bbox.top + obj_meta.bbox.height / 2),
                    )
                    # self.track_last_frame_num[track_id] = nvds_frame_meta.frame_num
                    direction = lc_tracker.check_track(obj_meta.track_id)

                    obj_events = self.cross_events[frame_meta.source_id][obj_meta.track_id]
                    if direction is not None:
                        obj_events.append((direction.name, frame_meta.pts))
                        
                        if direction == Direction.entry:
                            self.entry_count[frame_meta.source_id] += 1
                        elif direction == Direction.exit:
                            self.exit_count[frame_meta.source_id] += 1

                    for direction_name, frame_pts in obj_events:
                        obj_meta.add_attr_meta('lc_tracker', direction_name, frame_pts)

            primary_meta_object.add_attr_meta(
                'analytics', 'entries_n', self.entry_count[frame_meta.source_id]
            )
            primary_meta_object.add_attr_meta(
                'analytics', 'exits_n', self.exit_count[frame_meta.source_id]
            )
            primary_meta_object.add_attr_meta(
                'analytics', 'line_from', line_from
            )
            primary_meta_object.add_attr_meta(
                'analytics', 'line_to', line_to
            )


