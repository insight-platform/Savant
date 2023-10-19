import pyds

from savant.deepstream.meta.frame import NvDsFrameMeta
from savant.deepstream.pyfunc import NvDsPyFuncPlugin
from savant.gstreamer import Gst  # noqa: F401


class StatLogger(NvDsPyFuncPlugin):
    """Helper to log stats for run_perf script.
    TODO: Add fps measurement 1) per source and 2) with "padding"
        (to avoid the influence of slow start and delay in forming the last batch)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.counters = {
            'num_frames_in_batch': [],
            'num_frames_per_source': {},
            'num_objects_per_source': {},
        }

    def on_stop(self) -> bool:
        """Do on plugin stop."""
        log_msgs = []

        # the last batch can differ (depends on video length)
        num_frames_in_batch = self.counters['num_frames_in_batch'][:-1]
        if num_frames_in_batch:
            avg = sum(num_frames_in_batch) / len(num_frames_in_batch)
            log_msgs.append(
                f'num_frames_in_batch: '
                f'min={min(num_frames_in_batch)}, '
                f'max={max(num_frames_in_batch)}, '
                f'avg={avg:.2f}'
            )

        num_frames = self.counters['num_frames_per_source']
        if num_frames:
            for source_id in sorted(num_frames):
                log_msgs.append(f'num_frames[{source_id}]: {num_frames[source_id]}')
            avg = sum(num_frames.values()) / len(num_frames)
            log_msgs.append(
                f'num_frames_per_source: '
                f'min={min(num_frames.values())}, '
                f'max={max(num_frames.values())}, '
                f'avg={avg:.2f}'
            )

        num_objects = self.counters['num_objects_per_source']
        if num_objects:
            for source_id in sorted(num_objects):
                log_msgs.append(f'num_objects[{source_id}]: {num_objects[source_id]}')
            avg = sum(num_objects.values()) / len(num_objects)
            log_msgs.append(
                f'num_objects_per_source: '
                f'min={min(num_objects.values())}, '
                f'max={max(num_objects.values())}, '
                f'avg={avg:.2f}'
            )

        if not log_msgs:
            log_msgs.append('something went wrong..')
        self.logger.info('Stats\n' + '\n'.join(log_msgs))

        return super().on_stop()

    def on_source_add(self, source_id: str):
        """On source add event callback."""
        self.counters['num_frames_per_source'][source_id] = 0
        self.counters['num_objects_per_source'][source_id] = 0

    def process_buffer(self, buffer: Gst.Buffer):
        """Process gstreamer buffer directly."""
        nvds_batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(buffer))
        self.counters['num_frames_in_batch'].append(nvds_batch_meta.num_frames_in_batch)
        super().process_buffer(buffer)

    def process_frame(self, buffer: Gst.Buffer, frame_meta: NvDsFrameMeta):
        self.counters['num_frames_per_source'][frame_meta.source_id] += 1
        # count objects without primary object (frame)
        for obj_meta in frame_meta.objects:
            if not obj_meta.is_primary:
                self.counters['num_objects_per_source'][frame_meta.source_id] += 1
