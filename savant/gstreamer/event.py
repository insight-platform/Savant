from typing import Optional

from savant.gstreamer import Gst

SAVANT_EOS_EVENT_NAME = 'savant-eos'
SAVANT_EOS_EVENT_SOURCE_ID_PROPERTY = 'source-id'


def build_savant_eos_event(source_id: str):
    """Build a savant-eos event.

    :param source_id: Source ID of the stream.
    :returns: The savant-eos event.
    """

    structure: Gst.Structure = Gst.Structure.new_empty(SAVANT_EOS_EVENT_NAME)
    structure.set_value(SAVANT_EOS_EVENT_SOURCE_ID_PROPERTY, source_id)
    return Gst.Event.new_custom(Gst.EventType.CUSTOM_DOWNSTREAM, structure)


def parse_savant_eos_event(event: Gst.Event) -> Optional[str]:
    """Parse a savant-eos event.

    :param event: The event to parse.
    :returns: Source ID of the stream if the event is a savant-eos event, otherwise None.
    """

    if event.type != Gst.EventType.CUSTOM_DOWNSTREAM:
        return None

    struct: Gst.Structure = event.get_structure()
    if not struct.has_name(SAVANT_EOS_EVENT_NAME):
        return None

    return struct.get_string(SAVANT_EOS_EVENT_SOURCE_ID_PROPERTY)
