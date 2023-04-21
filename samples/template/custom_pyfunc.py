"""Custom PyFunc implementation."""
from savant.gstreamer import Gst
from savant.deepstream.meta.frame import NvDsFrameMeta
from savant.deepstream.pyfunc import NvDsPyFuncPlugin


class CustomPyFunc(NvDsPyFuncPlugin):
    """..."""

    def process_frame(self, buffer: Gst.Buffer, frame_meta: NvDsFrameMeta):
        """...

        :param buffer: GStreamer buffer.
        :param frame_meta: Processed frame metadata.
        """
        self.logger.info(
            'Processing frame #%d of source %s..',
            frame_meta.frame_num,
            frame_meta.source_id,
        )
