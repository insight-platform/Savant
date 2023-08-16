"""Base NvDsPyFuncPlugin for drawing on frame."""

from abc import abstractmethod
from typing import Any, Dict

from savant.config.schema import FrameProcessingCondition
from savant.deepstream.meta.frame import NvDsFrameMeta
from savant.deepstream.pyfunc import NvDsPyFuncPlugin
from savant.gstreamer import Gst  # noqa: F401


class BaseNvDsDrawFunc(NvDsPyFuncPlugin):
    """Base PyFunc for drawing on frame.

    PyFunc implementations are defined in and instantiated by a
    :py:class:`.PyFunc` structure.

    :param condition: Conditions for filtering frames to be processed by the
        draw function. The draw function will be applied only to frames when
        all conditions are met.
    """

    def __init__(self, condition: Dict[str, Any], **kwargs):
        self.condition = FrameProcessingCondition(**condition)
        super().__init__(**kwargs)

    @abstractmethod
    def draw(self, buffer: Gst.Buffer, frame_meta: NvDsFrameMeta):
        """Draw metadata on a frame in a batch.

        :param buffer: Gstreamer buffer.
        :param frame_meta: Frame metadata for a frame in a batch.
        """

    def process_frame(self, buffer: Gst.Buffer, frame_meta: NvDsFrameMeta):
        if self.can_draw_on_frame(frame_meta):
            self.draw(buffer, frame_meta)

    def can_draw_on_frame(self, frame_meta: NvDsFrameMeta) -> bool:
        """Check whether we can draw on this specific frame."""

        if self.condition.tag is None:
            return True

        if self.condition.tag in frame_meta.tags:
            return True

        self.logger.debug(
            'Frame from source %s with PTS %s does not have tag %s. Skip drawing on it.',
            frame_meta.source_id,
            frame_meta.pts,
            self.condition.tag,
        )
        return False
