"""Custom PyFunc implementation."""

from savant.deepstream.meta.frame import NvDsFrameMeta
from savant.deepstream.pyfunc import NvDsPyFuncPlugin
from savant.gstreamer import Gst


class CustomPyFunc(NvDsPyFuncPlugin):
    """Custom frame processor."""

    def process_frame(self, buffer: Gst.Buffer, frame_meta: NvDsFrameMeta):
        """Process frame.

        :param buffer: GStreamer buffer.
        :param frame_meta: Processed frame metadata.
        """
        self.logger.info(
            'Processing frame #%d of source %s..',
            frame_meta.frame_num,
            frame_meta.source_id,
        )

    def on_source_eos(self, source_id: str):
        """On source EOS event callback."""
        self.logger.debug('Got GST_NVEVENT_STREAM_EOS for source %s.', source_id)
