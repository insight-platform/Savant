"""Example of a Python plugin that prints the output to the log."""

from savant.deepstream.meta.frame import NvDsFrameMeta
from savant.deepstream.pyfunc import NvDsPyFuncPlugin
from savant.gstreamer import Gst
from savant.parameter_storage import param_storage

MODEL_NAME = param_storage()['model_name']
ATTR_NAME = param_storage()['attribute_name']


class PrintOutput(NvDsPyFuncPlugin):
    """Print output to log."""

    def process_frame(self, buffer: Gst.Buffer, frame_meta: NvDsFrameMeta):
        """Process frame metadata.

        :param buffer: Gstreamer buffer with this frame's data.
        :param frame_meta: This frame's metadata.
        """
        attr = None
        for obj_meta in frame_meta.objects:
            if obj_meta.is_primary:
                attr = obj_meta.get_attr_meta(MODEL_NAME, ATTR_NAME)
                break

        if attr:
            self.logger.info('Pre-processed data %s', attr.value)
