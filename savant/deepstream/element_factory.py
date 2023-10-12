"""DeepStream element factory."""
from dataclasses import replace

import pyds

from savant.config.schema import PipelineElement
from savant.gstreamer import Gst  # noqa: F401
from savant.gstreamer.element_factory import GstElementFactory
from savant.utils.platform import is_aarch64


class NvDsElementFactory(GstElementFactory):
    """Creates elements for DeepStream pipeline."""

    def create(self, element: PipelineElement) -> Gst.Element:
        """Creates specified element.

        :param element: Pipeline element to create.
        :raises CreateElementException: Unknown element.
        :return: Gst.Element.
        """

        if element.element == 'nvvideoconvert':
            return self.create_nvvideoconvert(element)

        return super().create(element)

    @staticmethod
    def create_nvvideoconvert(element: PipelineElement) -> Gst.Element:
        """Creates nvvideoconvert element.

        Default properties are changed based on platform.
        """

        element = replace(
            element,
            properties={
                **NvDsElementFactory.default_nvvideoconvert_properties(),
                **element.properties,
            },
        )

        return GstElementFactory.create_element(element)

    @staticmethod
    def default_nvvideoconvert_properties():
        """Get default nvvideoconvert properties based on platform."""

        if is_aarch64():
            return {'copy-hw': 2}  # VIC
        else:
            return {'nvbuf-memory-type': int(pyds.NVBUF_MEM_CUDA_UNIFIED)}
