"""GStreamer pipeline elements factory."""

from typing import Union

from gi.repository import Gst  # noqa:F401

from savant.base.model import AttributeModel, ComplexModel, ObjectModel
from savant.config.schema import ModelElement, PipelineElement


class CreateElementException(Exception):
    """Unable to create Gst.Element Exception."""


class GstElementFactory:
    """Creates pipeline elements."""

    def create(self, element: PipelineElement) -> Gst.Element:
        """Creates specified element.

        :param element: Pipeline element to create.
        :raises CreateElementException: Unknown element.
        :return: Gst.Element.
        """
        if element.element == 'capsfilter':
            return self.create_caps_filter(element)

        if element.element == 'videotestsrc':
            return self.create_videotestsrc(element)

        if isinstance(element, ModelElement):
            return self.create_model_element(element)

        if isinstance(element, PipelineElement):
            return self.create_element(element)

        raise CreateElementException(
            f'Undefined element {type(element)} {element} to create.'
        )

    @staticmethod
    def create_element(element: PipelineElement) -> Gst.Element:
        """Creates Gst.Element.

        :param element: PipelineElement to create.
        :raises CreateElementException: Unable to create element.
        :return: Created Gst.Element
        """
        gst_element = Gst.ElementFactory.make(element.element, element.name)
        if not gst_element:
            raise CreateElementException(f'Unable to create element {element}.')

        # set element name from GstElement
        if element.name is None:
            element.name = gst_element.name

        for prop_name, prop_value in element.properties.items():
            if prop_value is not None:
                gst_element.set_property(prop_name, prop_value)

        return gst_element

    @staticmethod
    def create_model_element(element: ModelElement) -> Gst.Element:
        """Creates Gst.Element for ModelElement.

        :param element: ModelElement to create.
        :return: Created Gst.Element
        """

        model: Union[AttributeModel, ComplexModel, ObjectModel] = element.model

        if model.input.preprocess_object_meta:
            model.input.preprocess_object_meta.load_user_code()
        if model.input.preprocess_object_image:
            model.input.preprocess_object_image.load_user_code()
        if model.output.converter:
            model.output.converter.load_user_code()
        if isinstance(model, (ObjectModel, ComplexModel)):
            for obj in model.output.objects:
                if obj.selector:
                    obj.selector.load_user_code()

        return GstElementFactory.create_element(element)

    @staticmethod
    def create_caps_filter(element: PipelineElement) -> Gst.Element:
        """Creates ``capsfilter`` Gst.Element.

        :param element: Element to create.
        :return: Created Gst.Element
        """
        caps = None
        if 'caps' in element.properties and isinstance(element.properties['caps'], str):
            caps = Gst.Caps.from_string(element.properties['caps'])
            del element.properties['caps']
        gst_element = GstElementFactory.create_element(element)
        if caps:
            gst_element.set_property('caps', caps)
        return gst_element

    @staticmethod
    def create_videotestsrc(element: PipelineElement) -> Gst.Bin:
        """Creates ``videotestsrc`` element as a Gst.Bin with ``pad-added``.

        :param element: Element to create.
        :return: Created Gst.Element
        """

        caps_filter = None
        if 'caps' in element.properties:
            caps_filter = GstElementFactory.create_caps_filter(
                PipelineElement(
                    'capsfilter',
                    properties={'caps': element.properties['caps']},
                )
            )
            del element.properties['caps']

        src_element = GstElementFactory.create_element(element)

        src_decodebin = Gst.Bin.new(element.name)

        Gst.Bin.add(src_decodebin, src_element)

        if caps_filter:
            Gst.Bin.add(src_decodebin, caps_filter)
            src_element.link(caps_filter)

        decodebin = GstElementFactory.create_element(PipelineElement('decodebin'))

        def on_pad_added(elem: Gst.Element, pad: Gst.Pad):
            """Proxy newly added pad to bin."""
            ghost_pad: Gst.GhostPad = Gst.GhostPad.new(pad.get_name(), pad)
            ghost_pad.set_active(True)
            src_decodebin.add_pad(ghost_pad)

        def on_pad_removed(elem: Gst.Element, pad: Gst.Pad):
            """Remove ghost pad for removed pad."""
            for ghost_pad in src_decodebin.iterate_pads():
                if ghost_pad.get_name() == pad.get_name():
                    src_decodebin.remove_pad(ghost_pad)
                    return

        decodebin.connect('pad-added', on_pad_added)
        decodebin.connect('pad-removed', on_pad_removed)

        Gst.Bin.add(src_decodebin, decodebin)

        if caps_filter:
            caps_filter.link(decodebin)
        else:
            src_element.link(decodebin)

        return src_decodebin
