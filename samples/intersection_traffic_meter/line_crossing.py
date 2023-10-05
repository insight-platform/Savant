import sys
from collections import defaultdict
from itertools import permutations
import yaml
from savant_rs.primitives.geometry import Point, PolygonalArea
from statsd import StatsClient

from samples.intersection_traffic_meter.utils import Point, TwoLinesCrossingTracker
from savant.deepstream.meta.frame import NvDsFrameMeta
from savant.deepstream.pyfunc import NvDsPyFuncPlugin
from savant.gstreamer import Gst


class ConditionalDetectorSkip(NvDsPyFuncPlugin):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        with open(self.config_path, 'r', encoding='utf8') as stream:
            self.line_config = yaml.safe_load(stream)

    def process_frame(self, buffer: Gst.Buffer, frame_meta: NvDsFrameMeta):
        primary_meta_object = None
        for obj_meta in frame_meta.objects:
            if obj_meta.is_primary:
                primary_meta_object = obj_meta
                break

        # if the boundary lines are not configured for this source
        # then disable detector inference entirely by removing the primary object
        # Note:
        # In order to enable use cases such as conditional inference skip
        # or user-defined ROI, Savant configures all Deepstream models to run
        # in 'secondary' mode and inserts a primary 'frame' object into the DS meta
        if (
            primary_meta_object is not None
            and frame_meta.source_id not in self.line_config
        ):
            frame_meta.remove_obj_meta(primary_meta_object)


class LineCrossing(NvDsPyFuncPlugin):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        with open(self.config_path, 'r', encoding='utf8') as stream:
            self.line_config = yaml.safe_load(stream)

        self.areas = {}
        for source_id, poly_cfg in self.line_config.items():
            # The conversion from 2 lines to a 4 point polygon is as follows:
            # assuming the lines are AB and CD, the polygon is ABDC.
            # The AB polygon edge is marked as "from" and the CD edge is marked as "to".
            points = [
                Point(*coords)
                for coords in poly_cfg['points']
            ]

            success = False
            for points_perm in permutations(points):
                area = PolygonalArea(points_perm, poly_cfg['edges'])
                if not area.is_self_intersecting():
                    success = True
                    break

            if not success:
                self.logger.error(
                    'Lines config for the "%s" source id produced a self-intersecting polygon.'
                    ' Please correct coordinates "%s" in the config file and restart the pipeline.',
                    source_id,
                    poly_cfg,
                )
                sys.exit(1)
            self.areas[source_id] = area

        self.lc_trackers = {}
        self.track_last_frame_num = defaultdict(lambda: defaultdict(int))
        self.crossing_counts = defaultdict(lambda: defaultdict(int))

        self.cross_events = defaultdict(lambda: defaultdict(list))

        # metrics namescheme
        # savant.module.intersection_traffic_meter.source_id.obj_class_label.crossing_label
        if self.send_stats:
            self.stats_client = StatsClient(
                'graphite', 8125, prefix='savant.module.intersection_traffic_meter'
            )

    def on_source_eos(self, source_id: str):
        """On source EOS event callback."""
        if source_id in self.lc_trackers:
            del self.lc_trackers[source_id]
        if source_id in self.track_last_frame_num:
            del self.track_last_frame_num[source_id]
        if source_id in self.cross_events:
            del self.cross_events[source_id]
        if source_id in self.crossing_counts:
            del self.crossing_counts[source_id]


    def process_frame(self, buffer: Gst.Buffer, frame_meta: NvDsFrameMeta):
        """Process frame metadata.

        :param buffer: Gstreamer buffer with this frame's data.
        :param frame_meta: This frame's metadata.
        """
        # the primary meta object may be missed in the first several frames
        # due to nvtracker deleting all unconfirmed tracks
        primary_meta_object = None
        for obj_meta in frame_meta.objects:
            if obj_meta.is_primary:
                primary_meta_object = obj_meta
                break

        if primary_meta_object is not None and frame_meta.source_id in self.areas:
            if frame_meta.source_id not in self.lc_trackers:
                self.lc_trackers[frame_meta.source_id] = TwoLinesCrossingTracker(
                    self.areas[frame_meta.source_id]
                )
            lc_tracker = self.lc_trackers[frame_meta.source_id]

            obj_metas = []
            for obj_meta in frame_meta.objects:
                if obj_meta.label == self.target_obj_label:
                    if not self.cross_events[frame_meta.source_id][obj_meta.track_id]:
                        lc_tracker.add_track_point(
                            obj_meta.track_id,
                            # center point
                            Point(
                                obj_meta.bbox.xc,
                                obj_meta.bbox.yc,
                            ),
                        )
                        self.track_last_frame_num[frame_meta.source_id][
                            obj_meta.track_id
                        ] = frame_meta.frame_num

                        obj_metas.append(obj_meta)

            track_lines_crossings = lc_tracker.check_tracks(
                [obj_meta.track_id for obj_meta in obj_metas]
            )

            for obj_meta, cross_result in zip(obj_metas, track_lines_crossings):
                obj_events = self.cross_events[frame_meta.source_id][obj_meta.track_id]

                if cross_result is not None:

                    if '->' in cross_result:
                        # full crossing and not just entry
                        if self.send_stats:
                            # send to graphite
                            target = '.'.join(
                                (
                                    frame_meta.source_id,
                                    self.target_obj_label,
                                    cross_result.replace('->', '_'),
                                )
                            )
                            self.logger.debug('Incrementing metric %s', target)
                            self.stats_client.incr(target)
                        obj_events.append((cross_result, frame_meta.pts))

                    self.crossing_counts[frame_meta.source_id][cross_result] += 1

            for obj_meta in frame_meta.objects:
                obj_events = self.cross_events[frame_meta.source_id][obj_meta.track_id]
                for direction_name, frame_pts in obj_events:
                    obj_meta.add_attr_meta('lc_tracker', direction_name, frame_pts)

            for direction, crossings_n in self.crossing_counts[frame_meta.source_id].items():
                primary_meta_object.add_attr_meta(
                    'analytics', direction, crossings_n
                )


        # periodically remove stale tracks
        if not (frame_meta.frame_num % self.stale_track_del_period):
            last_frames = self.track_last_frame_num[frame_meta.source_id]

            to_delete = [
                track_id
                for track_id, last_frame in last_frames.items()
                if frame_meta.frame_num - last_frame > self.stale_track_del_period
            ]
            if to_delete:
                for track_id in to_delete:
                    lc_tracker = self.lc_trackers[frame_meta.source_id]
                    del last_frames[track_id]
                    lc_tracker.remove_track(track_id)
