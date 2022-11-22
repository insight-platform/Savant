"""DeepStream pipeline elements factory."""
from dataclasses import replace
import pyds
from savant.config.schema import PipelineElement, PyFuncElement
from savant.gstreamer import Gst  # noqa:F401
from savant.gstreamer.element_factory import GstElementFactory
from savant.utils.platform import is_aarch64


class NvDsElementFactory(GstElementFactory):
    """Creates pipeline elements."""

    def create(self, element: PipelineElement) -> Gst.Element:
        """Creates specified element.

        :param element: Pipeline element to create.
        :raises CreateElementException: Unknown element.
        :return: Gst.Element.
        """
        # if isinstance(element, PyFuncElement):
        #     return self.create_pyfunc_bin(element)
        return super().create(element)

    @staticmethod
    def create_pyfunc_bin(element: PyFuncElement) -> Gst.Element:
        """Creates Gst.Element.

        :param element: pipeline element to create.
        """

        pyfunc_bin = Gst.Bin.new(element.name)

        queue: Gst.Element = Gst.ElementFactory.make('queue', 'pyfunc_queue')
        pyfunc_bin.add(queue)

        conv: Gst.Element = Gst.ElementFactory.make('nvvideoconvert', 'pyfunc_conv')
        if not is_aarch64():
            conv.set_property('nvbuf-memory-type', int(pyds.NVBUF_MEM_CUDA_UNIFIED))
        pyfunc_bin.add(conv)

        conv_queue: Gst.Element = Gst.ElementFactory.make('queue', 'pyfunc_conv_queue')
        pyfunc_bin.add(conv_queue)

        pyfunc: Gst.Element = GstElementFactory.create_element(
            replace(element, name='pyfunc_pyfunc')
        )
        pyfunc_bin.add(pyfunc)

        assert queue.link(conv)
        assert conv.link(conv_queue)
        assert conv_queue.link(pyfunc)

        sink_pad = Gst.GhostPad.new('sink', queue.get_static_pad('sink'))
        pyfunc_bin.add_pad(sink_pad)

        src_pad = Gst.GhostPad.new('src', pyfunc.get_static_pad('src'))
        pyfunc_bin.add_pad(src_pad)

        return pyfunc_bin
