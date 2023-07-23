"""GStreamer pipeline elements factory."""
import dataclasses
from gi.repository import Gst  # noqa:F401
from savant.config.schema import PipelineElement, ModelElement, PyFuncElement
from savant.gstreamer.element_factory import GstElementFactory


class NvDsElementFactory(GstElementFactory):
    """Creates pipeline elements."""

    def create(self, element: PipelineElement) -> Gst.Element:
        """Creates specified element.

        :param element: Pipeline element to create.
        :raises CreateElementException: Unknown element.
        :return: Gst.Element.
        """
        if element.element == 'nvinfer':
            return self.create_nvinfer(element)

        return super().create(element)

    def create_nvinfer(self, element: ModelElement) -> Gst.Element:
        """NvInfer element as Gst.Bin with queue and pyfunc postprocessor."""
        infer_bin = Gst.Bin.new(element.name)

        # pipeline with uridecodebin source hangs with queue before nvinfer,
        # so we add queue only before pyfunc

        infer_element = dataclasses.replace(element, name=f'{element.name}_infer')
        infer = self.create_element(infer_element)
        infer_bin.add(infer)

        pyfunc_queue = Gst.ElementFactory.make('queue', f'{element.name}_pyfunc_queue')
        infer_bin.add(pyfunc_queue)

        pyfunc = self.create(
            PyFuncElement(
                name=f'{element.name}_pyfunc',
                module='savant.deepstream.nvinfer_postproc',
                class_name='NvInferPostprocessor',
                properties={
                    'pyobject': element,
                },
            )
        )
        infer_bin.add(pyfunc)

        assert infer.link(pyfunc_queue)
        assert pyfunc_queue.link(pyfunc)

        sink_pad = infer.get_static_pad('sink')
        infer_bin.add_pad(Gst.GhostPad.new('sink', sink_pad))

        src_pad = pyfunc.get_static_pad('src')
        infer_bin.add_pad(Gst.GhostPad.new('src', src_pad))

        return infer_bin
