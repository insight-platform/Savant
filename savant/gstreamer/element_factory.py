"""GStreamer pipeline elements factory."""

from typing import Dict, List, Union

from gi.repository import GLib, Gst  # noqa:F401
from savant_rs.py.api.constants import DEFAULT_FRAMERATE

from savant.base.model import AttributeModel, ComplexModel, ObjectModel
from savant.config.schema import ModelElement, PipelineElement
from savant.utils.platform import is_aarch64


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

        if element.element == 'nvarguscamerasrc_bin':
            return self.create_nvarguscamerasrc_bin(element)

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

    @staticmethod
    def create_nvarguscamerasrc_bin(element: PipelineElement) -> Gst.Bin:
        """Creates ``nvarguscamerasrc_bin`` element as a Gst.Bin wrapping
        multiple ``nvarguscamerasrc`` elements.

        Example::

            element: nvarguscamerasrc_bin
            properties:
              sources:
                - source-id: camera-0
                  framerate: 25/1
                  properties:
                    sensor-id: 0
                - source-id: camera-1
                  framerate: 10/1
                  properties:
                    sensor-id: 1
                    sensor-mode: 1

        :param element: Element to create.
        :return: Created Gst.Element
        """

        if not is_aarch64():
            raise CreateElementException(
                'nvarguscamerasrc_bin: only supported on aarch64.'
            )
        if 'sources' not in element.properties:
            raise CreateElementException(
                'nvarguscamerasrc_bin: sources list must be specified.'
            )
        sources: List[Dict] = element.properties['sources']
        if not isinstance(sources, list) or not sources:
            raise CreateElementException(
                'nvarguscamerasrc_bin: sources must be non-empty list.'
            )

        source_ids = [source_cfg.get('source-id') for source_cfg in sources]
        if any(sid is None for sid in source_ids):
            raise CreateElementException(
                'nvarguscamerasrc_bin: source-id is required for each source.'
            )
        if len(source_ids) != len(set(source_ids)):
            raise CreateElementException(
                'nvarguscamerasrc_bin: source-id values must be unique.'
            )

        src_bin = Gst.Bin.new(element.name)
        ghost_pads: List[Gst.GhostPad] = []

        for source_cfg in sources:
            source_id = source_cfg['source-id']
            framerate = source_cfg.get('framerate', DEFAULT_FRAMERATE)
            source_properties = source_cfg.get('properties', {})

            src_element = Gst.ElementFactory.make(
                'nvarguscamerasrc', f'argus_{source_id}'
            )
            if not src_element:
                raise CreateElementException(
                    f'nvarguscamerasrc_bin: failed to create nvarguscamerasrc '
                    f'for source {source_id}. '
                    f'Is the nvarguscamerasrc plugin available?'
                )
            if source_properties:
                for prop_name, prop_value in source_properties.items():
                    if prop_value is not None:
                        src_element.set_property(prop_name, prop_value)

            caps_str = f'video/x-raw(memory:NVMM), framerate={framerate}'
            caps_filter = Gst.ElementFactory.make(
                'capsfilter', f'argus_caps_{source_id}'
            )
            if not caps_filter:
                raise CreateElementException(
                    'nvarguscamerasrc_bin: failed to create capsfilter.'
                )
            caps_filter.set_property('caps', Gst.Caps.from_string(caps_str))

            src_bin.add(src_element)
            src_bin.add(caps_filter)
            src_element.link(caps_filter)

            caps_src_pad = caps_filter.get_static_pad('src')
            caps_src_pad.add_probe(
                Gst.PadProbeType.EVENT_UPSTREAM,
                drop_reconfigure_event,
            )
            ghost_pad = Gst.GhostPad.new(f'src_{source_id}', caps_src_pad)
            ghost_pads.append(ghost_pad)

        def add_src_pads_to_bin(*args):
            """Add ghost pads to bin asynchronously."""
            for ghost_pad in ghost_pads:
                ghost_pad.set_active(True)
                src_bin.add_pad(ghost_pad)
            return False

        GLib.idle_add(add_src_pads_to_bin)

        return src_bin


def drop_reconfigure_event(pad: Gst.Pad, info: Gst.PadProbeInfo) -> Gst.PadProbeReturn:
    """Pad probe callback that drops upstream RECONFIGURE events.

    Prevents nvarguscamerasrc from renegotiating caps and reopening a session when
    nvstreammux propagates batch-size/num-surfaces-per-frame upstream.
    """
    event: Gst.Event = info.get_event()
    if event is not None and event.type == Gst.EventType.RECONFIGURE:
        return Gst.PadProbeReturn.DROP
    return Gst.PadProbeReturn.OK
