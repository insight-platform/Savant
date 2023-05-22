from collections import defaultdict
import yaml
from statsd import StatsClient
from savant.gstreamer import Gst
from savant.deepstream.meta.frame import NvDsFrameMeta
from savant.deepstream.pyfunc import NvDsPyFuncPlugin
from samples.traffic_meter.utils import TwoLinesCrossingTracker, Point, Direction
from timeit import default_timer as timer
import numpy as np
class ConditionalDetectorSkip(NvDsPyFuncPlugin):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        with open(self.config_path, "r", encoding='utf8') as stream:
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
        with open(self.config_path, "r", encoding='utf8') as stream:
            self.line_config = yaml.safe_load(stream)

        self.lc_trackers = {}
        self.track_last_frame_num = defaultdict(lambda: defaultdict(int))
        self.entry_count = defaultdict(int)
        self.exit_count = defaultdict(int)
        self.cross_events = defaultdict(lambda: defaultdict(list))

        # metrics namescheme
        # savant.module.traffic_meter.source_id.obj_class_label.exit
        # savant.module.traffic_meter.source_id.obj_class_label.entry
        # self.stats_client = StatsClient(
        #     'graphite', 8125, prefix='savant.module.traffic_meter'
        # )

        self.per_obj_times_py = []
        self.per_obj_times_rs = []
        self.per_frame_times_py = []
        self.times_rs_batch = []

    def on_source_eos(self, source_id: str):
        """On source EOS event callback."""
        if source_id in self.lc_trackers:
            del self.lc_trackers[source_id]
        if source_id in self.track_last_frame_num:
            del self.track_last_frame_num[source_id]
        if source_id in self.cross_events:
            del self.cross_events[source_id]
        if source_id in self.entry_count:
            del self.entry_count[source_id]
        if source_id in self.exit_count:
            del self.exit_count[source_id]

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

        if primary_meta_object is not None and frame_meta.source_id in self.line_config:
            line_from = self.line_config[frame_meta.source_id]['from']
            line_to = self.line_config[frame_meta.source_id]['to']

            if frame_meta.source_id not in self.lc_trackers:
                self.lc_trackers[frame_meta.source_id] = TwoLinesCrossingTracker(
                    (Point(*line_from[:2]), Point(*line_from[2:])),
                    (Point(*line_to[:2]), Point(*line_to[2:])),
                )
            lc_tracker = self.lc_trackers[frame_meta.source_id]

            track_ids = []
            py_checks = []
            rs_checks = []
            this_frame_per_obj_times_py = []
            for obj_meta in frame_meta.objects:
                if obj_meta.label == self.target_obj_label:
                    lc_tracker.add_track_point(
                        obj_meta.track_id,
                        # center point
                        Point(
                            obj_meta.bbox.left + obj_meta.bbox.width / 2,
                            obj_meta.bbox.top + obj_meta.bbox.height / 2,
                        ),
                    )
                    self.track_last_frame_num[frame_meta.source_id][
                        obj_meta.track_id
                    ] = frame_meta.frame_num
                    track_ids.append(obj_meta.track_id)

                    t1 = timer()
                    direction_py = lc_tracker.check_track(obj_meta.track_id)
                    t2 = timer()
                    direction_rs = lc_tracker.check_track_rs(obj_meta.track_id)
                    t3 = timer()
                    self.per_obj_times_py.append(t2-t1)
                    this_frame_per_obj_times_py.append(t2-t1)
                    self.per_obj_times_rs.append(t3-t2)
                    py_checks.append(direction_py)
                    rs_checks.append(direction_rs)

                    obj_events = self.cross_events[frame_meta.source_id][
                        obj_meta.track_id
                    ]
                    if direction_py is not None:
                        # send to graphite
                        # self.stats_client.incr(
                        #     '.'.join(
                        #         (
                        #             frame_meta.source_id,
                        #             self.target_obj_label,
                        #             direction.name,
                        #         )
                        #     )
                        # )

                        obj_events.append((direction_py.name, frame_meta.pts))

                        if direction_py == Direction.entry:
                            self.entry_count[frame_meta.source_id] += 1
                        elif direction_py == Direction.exit:
                            self.exit_count[frame_meta.source_id] += 1

                    for direction_name, frame_pts in obj_events:
                        obj_meta.add_attr_meta('lc_tracker', direction_name, frame_pts)

            self.per_frame_times_py.append(sum(this_frame_per_obj_times_py))

            t4 = timer()
            batch_rs_checks = lc_tracker.check_track_rs_batch(track_ids)
            self.times_rs_batch.append(timer()-t4)

            assert len(py_checks) == len(rs_checks) == len(batch_rs_checks)
            for py_check, rs_check, b_rs_check in zip(py_checks, rs_checks, batch_rs_checks):
                assert py_check == rs_check == b_rs_check

            primary_meta_object.add_attr_meta(
                'analytics', 'entries_n', self.entry_count[frame_meta.source_id]
            )
            primary_meta_object.add_attr_meta(
                'analytics', 'exits_n', self.exit_count[frame_meta.source_id]
            )
            primary_meta_object.add_attr_meta('analytics', 'line_from', line_from)
            primary_meta_object.add_attr_meta('analytics', 'line_to', line_to)

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

    def on_stop(self):
        print(f"avg per obj py: {np.mean(self.per_obj_times_py)*10**6:.3f}, avg per obj rs: {np.mean(self.per_obj_times_rs)*10**6:.3f}")
        print(f"avg delta rs: {(np.mean(np.abs(np.array(self.per_obj_times_py)-np.array(self.per_obj_times_rs))))*10**6:.3f}")

        print(f"avg per frame py {np.mean(self.per_frame_times_py)*10**6:.3f}, avg per frame rs: {np.mean(self.times_rs_batch)*10**6:.3f}")
        print(f"avg delta b_rs: {(np.mean(np.abs(np.array(self.per_frame_times_py)-np.array(self.times_rs_batch))))*10**6:.3f}")

        return True
