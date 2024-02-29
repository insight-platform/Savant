"""DeepStream pipeline runner class."""
from gi.repository import GLib, Gst  # noqa:F401

from savant.gstreamer.codecs import Codec
from savant.gstreamer.runner import GstPipelineRunner


class NvDsPipelineRunner(GstPipelineRunner):
    """Manages running DeepStream pipeline.

    :param pipeline: GstPipeline or Gst.Pipeline to run.
    """

    def build_error_message(self, message: Gst.Message, err: GLib.GError, debug: str):
        """Build error message."""
        error = super().build_error_message(message, err, debug)
        if (
            err.args
            and debug is not None
            and 'Device is in streaming mode' in debug
            and (
                message.src.get_factory().get_name()
                in [Codec.H264.value.nv_encoder, Codec.HEVC.value.nv_encoder]
            )
        ):
            return f'{error} Reached the limit of encoders sessions.'
        return error
