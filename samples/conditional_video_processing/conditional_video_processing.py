from typing import Dict, List, Set

from savant.deepstream.meta.frame import NvDsFrameMeta
from savant.deepstream.pyfunc import NvDsPyFuncPlugin
from savant.gstreamer import Gst


class ConditionalVideoProcessing(NvDsPyFuncPlugin):
    """Tags frames when they have specified detections.."""

    def __init__(
        self,
        detections: List[Dict],
        set_tags: List[str],
        protection_interval_ms: int,
        **kwargs,
    ):
        self.detections: Dict[str, Set[str]] = {
            d['element_name']: set(d['labels']) for d in detections
        }
        self.set_tags = {t: True for t in set_tags}
        self.protection_interval_ns = protection_interval_ms * Gst.MSECOND
        self.last_detections: Dict[str, int] = {}
        super().__init__(**kwargs)

    def process_frame(self, buffer: Gst.Buffer, frame_meta: NvDsFrameMeta):
        has_detections = any(
            obj.label in self.detections.get(obj.element_name, [])
            for obj in frame_meta.objects
        )
        self.logger.debug(
            'Frame with PTS %s has %s detections.',
            frame_meta.pts,
            'no' if not has_detections else '',
        )
        if has_detections:
            self.last_detections[frame_meta.source_id] = frame_meta.pts
        else:
            last_pts = self.last_detections.get(frame_meta.source_id)
            if (
                last_pts is None
                or frame_meta.pts - last_pts >= self.protection_interval_ns
            ):
                self.logger.debug('Skipping frame with PTS %s.', frame_meta.pts)
                if last_pts is not None:
                    del self.last_detections[frame_meta.source_id]
                return
        self.set_frame_tag(frame_meta)

    def on_source_eos(self, source_id: str):
        self.logger.info('Got EOS from source %s.', source_id)
        try:
            del self.last_detections[source_id]
        except KeyError:
            pass

    def set_frame_tag(self, frame_meta: NvDsFrameMeta):
        self.logger.debug(
            'Setting tags %s for frame %s/%s.',
            self.set_tags,
            frame_meta.source_id,
            frame_meta.pts,
        )
        for k, v in self.set_tags.items():
            frame_meta.set_tag(k, v)
