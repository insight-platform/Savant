"""GStreamer pipeline elements factory."""
from typing import Any
from gi.repository import Gst  # noqa:F401
from savant.config.schema import PipelineElement
from savant.parameter_storage import param_storage


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

        if isinstance(element, PipelineElement):
            return self.create_element(element)

        raise CreateElementException(
            f'Undefined element {type(element)} {element} to create.'
        )

    @staticmethod
    def create_element(element: PipelineElement) -> Gst.Element:
        """Creates Gst.Element.

        :param element: pipeline element to create.
        """
        gst_element = Gst.ElementFactory.make(element.element, element.name)
        if not gst_element:
            raise CreateElementException(f'Unable to create element {element}.')

        for prop_name, prop_value in element.properties.items():
            gst_element.set_property(prop_name, prop_value)

        for prop_name, dyn_gst_prop in element.dynamic_properties.items():

            def on_change(response_value: Any, property_name: str = prop_name):
                gst_element.set_property(property_name, response_value)

            param_storage().register_dynamic_parameter(
                dyn_gst_prop.storage_key, dyn_gst_prop.default, on_change
            )
            prop_value = param_storage()[dyn_gst_prop.storage_key]
            gst_element.set_property(prop_name, prop_value)

        return gst_element

    @staticmethod
    def create_caps_filter(element: PipelineElement) -> Gst.Element:
        caps = None
        if 'caps' in element.properties and isinstance(element.properties['caps'], str):
            caps = Gst.Caps.from_string(element.properties['caps'])
            del element.properties['caps']
        gst_element = GstElementFactory.create_element(element)
        if caps:
            gst_element.set_property('caps', caps)
        return gst_element
