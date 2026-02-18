"""Router ingress handler - creates left and right ROI objects on VideoFrame for each egress."""

from typing import Any

from savant_rs import register_handler
from savant_rs.logging import LogLevel, log
from savant_rs.match_query import MatchQuery
from savant_rs.primitives.geometry import RBBox
from savant_rs.utils.serialization import Message

from meta_merge.roi_constants import LEFT_ROI_LABEL, RIGHT_ROI_LABEL, ROI_NAMESPACE


class IngressHandler:
    """Creates top-level ROI objects (left half, right half) on VideoFrame for inference modules."""

    def __call__(
        self, message_id: int, ingress_name: str, topic: str, message: Message
    ) -> Message:
        """Add left and right ROI objects to VideoFrame so each module instance runs inference only on its ROI."""
        frame = message.as_video_frame()
        if frame is None:
            return message

        width = frame.width
        height = frame.height

        # Remove any existing primary/top-level objects (e.g. default full-frame ROI)
        frame.delete_objects(MatchQuery.not_(MatchQuery.parent_defined()))

        # Left half: x from 0 to width/2
        left_bbox = RBBox.ltwh(0, 0, width / 2, height)
        frame.create_object(
            namespace=ROI_NAMESPACE,
            label=LEFT_ROI_LABEL,
            detection_box=left_bbox,
        )

        # Right half: x from width/2 to width
        right_bbox = RBBox.ltwh(width / 2, 0, width / 2, height)
        frame.create_object(
            namespace=ROI_NAMESPACE,
            label=RIGHT_ROI_LABEL,
            detection_box=right_bbox,
        )

        log(
            LogLevel.Debug,
            "router",
            f"Added {ROI_NAMESPACE}.{LEFT_ROI_LABEL} and {ROI_NAMESPACE}.{RIGHT_ROI_LABEL} to frame {width}x{height}",
        )
        return message


def init(params: Any) -> bool:
    """Initialize router handlers."""
    register_handler("ingress_handler", IngressHandler())
    log(LogLevel.Info, "router", "Router initialized with ROI handler")
    return True
