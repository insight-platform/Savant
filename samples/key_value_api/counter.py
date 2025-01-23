import savant_rs.webserver.kvs as kvs
from savant_rs.primitives import Attribute, AttributeValue

from savant.deepstream.meta.frame import NvDsFrameMeta
from savant.deepstream.pyfunc import NvDsPyFuncPlugin
from savant.gstreamer import Gst


class Counter(NvDsPyFuncPlugin):
    """Apply gaussian blur to the frame."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._counter = 0

    def process_frame(self, buffer: Gst.Buffer, frame_meta: NvDsFrameMeta):
        self._counter += 1
        attr = Attribute(
            namespace='counter',
            name='frame_counter',
            hint='This attribute is set on every frame change',
            values=[
                AttributeValue.integer(self._counter),
            ],
        )
        kvs.set_attributes([attr], ttl=None)
