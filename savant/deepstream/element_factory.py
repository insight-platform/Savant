"""DeepStream element factory."""

from dataclasses import replace

import pyds
from pygstsavantframemeta import add_tracker_postproc_pad_probe

from savant.config.schema import PipelineElement
from savant.gstreamer import Gst  # noqa: F401
from savant.gstreamer.element_factory import GstElementFactory
from savant.utils.log import get_logger
from savant.utils.platform import is_aarch64

logger = get_logger(__name__)


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

        if element.element == 'nvtracker':
            return self.create_nvtracker(element)

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

    @staticmethod
    def create_nvtracker(element: PipelineElement) -> Gst.Element:
        """Creates nvtracker element with optional src pad probe
        that removes objects created by the tracker."""
        disable_obj_init = element.properties.pop('disable-obj-init', True)

        tracker = GstElementFactory.create_element(element)

        add_tracker_postproc_pad_probe(
            tracker.get_static_pad('src'),
            disable_obj_init=isinstance(disable_obj_init, bool) and disable_obj_init,
        )

        return tracker
