"""Scene change detector.

Reads the ReID "fingerprint" (embedding vector) computed for the ROI object and
compares it with a reference frame using cosine distance. The reference frame
is a frame ``reference_delay`` seconds ago.
"""

import collections

import numpy as np

from samples.scene_change.roi_injector import LABEL as ROI_LABEL
from savant.deepstream.meta.frame import NvDsFrameMeta
from savant.deepstream.pyfunc import NvDsPyFuncPlugin
from savant.gstreamer import Gst
from savant.parameter_storage import param_storage

# Element/attribute names written by this pyfunc (read by the overlay).
ELEMENT_NAME = 'scene_change'
ATTR_DISTANCE = 'distance'
ATTR_CHANGED = 'changed'


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine distance between two normalized vectors."""
    return 1.0 - float(np.dot(a, b))


class SceneChangeDetector(NvDsPyFuncPlugin):
    """Detects scene changes by comparing ROI fingerprints over time."""

    def __init__(
        self,
        dist_threshold: float = 0.2,
        reference_delay: float = 5.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dist_threshold = dist_threshold
        if reference_delay <= 0:
            raise ValueError(f'reference_delay must be > 0, got {reference_delay!r}.')
        self.reference_delay = reference_delay
        self.model_name = param_storage()['reid_model_name']
        self.history = collections.defaultdict(collections.deque)
        # To rate-limit logs
        self._changed_state = {}

    def process_frame(self, buffer: Gst.Buffer, frame_meta: NvDsFrameMeta):
        """Callback on each frame in a Deepstream pipeline batch."""
        roi_obj = None
        for obj_meta in frame_meta.objects:
            if obj_meta.label == ROI_LABEL:
                roi_obj = obj_meta
                break
        if roi_obj is None:
            return
        attr = roi_obj.get_attr_meta(self.model_name, 'reid')
        if attr is None:
            return

        current = np.asarray(attr.value, dtype=np.float32)
        norm = np.linalg.norm(current)
        if norm == 0:
            return

        current = current / norm
        source_id = frame_meta.source_id
        result = self._compare_delayed(source_id, frame_meta, current)
        if result is None:
            return

        distance, changed = result
        roi_obj.add_attr_meta(ELEMENT_NAME, ATTR_DISTANCE, distance)
        roi_obj.add_attr_meta(ELEMENT_NAME, ATTR_CHANGED, bool(changed))

        # Log only on the transition into a scene change, not every changed frame.
        if changed and not self._changed_state.get(source_id, False):
            self.logger.info(
                'Scene change detected for source %s (distance %.3f > %.3f).',
                source_id,
                distance,
                self.dist_threshold,
            )
        self._changed_state[source_id] = changed

    def _compare_delayed(self, source_id, frame_meta, current):
        """Compare the current frame against the reference."""

        pts = frame_meta.pts
        num, den = frame_meta.time_base
        if pts is None or den == 0:
            return None

        t_sec = pts * num / den
        target = t_sec - self.reference_delay
        buf = self.history[source_id]
        # Drop old entries
        while len(buf) >= 2 and buf[1][0] <= target:
            buf.popleft()
        buf.append((t_sec, current))

        reference = buf[0][1] if buf and buf[0][0] <= target else None
        if reference is None:
            # Not enough history yet.
            return None

        distance = cosine_distance(current, reference)
        changed = distance > self.dist_threshold

        return distance, changed

    def on_source_eos(self, source_id: str):
        self.history.pop(source_id, None)
        self._changed_state.pop(source_id, None)
