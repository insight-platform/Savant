"""Router ingress handler - creates left and right ROI objects on VideoFrame for each egress."""

import math
import random
from typing import Any

from savant_rs import register_handler
from savant_rs.logging import LogLevel, log
from savant_rs.match_query import MatchQuery
from savant_rs.primitives.geometry import RBBox
from savant_rs.utils.serialization import Message

from meta_merge.roi_constants import LEFT_ROI_LABEL, RIGHT_ROI_LABEL, ROI_NAMESPACE

MARKER_LABEL = 'marker'
MARKER_SIZE = 40


class IngressHandler:
    """Creates top-level ROI objects (left half, right half) on VideoFrame for inference modules."""

    def __init__(self):
        self._frame_idx = 0

    def __call__(
        self, message_id: int, ingress_name: str, topic: str, message: Message
    ) -> Message:
        """Add left and right ROI objects to VideoFrame so each module instance runs inference only on its ROI."""
        frame = message.as_video_frame()
        if frame is None:
            return message

        width = frame.width
        height = frame.height

        frame.export_complete_object_trees(MatchQuery.idle(), delete_exported=True)

        left_bbox = RBBox.ltwh(0, 0, width / 2, height)
        left_roi = frame.create_object(
            namespace=ROI_NAMESPACE,
            label=LEFT_ROI_LABEL,
            detection_box=left_bbox,
            confidence=1.0,
        )

        right_bbox = RBBox.ltwh(width / 2, 0, width / 2, height)
        right_roi = frame.create_object(
            namespace=ROI_NAMESPACE,
            label=RIGHT_ROI_LABEL,
            detection_box=right_bbox,
            confidence=1.0,
        )

        # Identical relative drift + confidence for both ROIs so users can match them.
        # Two-harmonic Lissajous for smooth, non-repeating motion.
        t = self._frame_idx * 0.02
        rel_x = 0.5 + 0.25 * math.sin(t) + 0.15 * math.sin(t * 1.7 + 0.3)
        rel_y = 0.5 + 0.25 * math.cos(t * 0.9) + 0.15 * math.cos(t * 1.3 + 0.7)
        conf = random.random()

        roi_w = width / 2
        roi_h = height
        pad = MARKER_SIZE / 2

        for roi, roi_left in ((left_roi, 0.0), (right_roi, roi_w)):
            cx = roi_left + pad + (roi_w - MARKER_SIZE) * rel_x
            cy = pad + (roi_h - MARKER_SIZE) * rel_y
            frame.create_object(
                namespace=ROI_NAMESPACE,
                label=MARKER_LABEL,
                detection_box=RBBox(cx, cy, MARKER_SIZE, MARKER_SIZE, 0.0),
                parent_id=roi.id,
                confidence=conf,
            )

        self._frame_idx += 1

        log(
            LogLevel.Debug,
            'router',
            f'Added ROIs + markers to frame {width}x{height} (idx={self._frame_idx})',
        )
        return message


def init(params: Any) -> bool:
    """Initialize router handlers."""
    register_handler('ingress_handler', IngressHandler())
    log(LogLevel.Info, 'router', 'Router initialized with ROI handler')
    return True
