from typing import Optional, Tuple

from savant.gstreamer import Gst

STREAM_PART_EVENT_NAME = 'stream-part'
STREAM_PART_EVENT_TAGGED_PROPERTY = 'tagged'
STREAM_PART_EVENT_PART_ID_PROPERTY = 'part-id'


def build_stream_part_event(part_id: int, tagged: bool):
    """Build a stream-part event with the given part id and tagged flag.

    :param part_id: ID of the stream part.
    :param tagged: Whether the stream part is tagged.
    :returns: The stream-part event.
    """

    structure: Gst.Structure = Gst.Structure.new_empty(STREAM_PART_EVENT_NAME)
    structure.set_value(STREAM_PART_EVENT_PART_ID_PROPERTY, part_id)
    structure.set_value(STREAM_PART_EVENT_TAGGED_PROPERTY, tagged)
    return Gst.Event.new_custom(Gst.EventType.CUSTOM_DOWNSTREAM, structure)


def parse_stream_part_event(event: Gst.Event) -> Optional[Tuple[int, bool]]:
    """Parse a stream-part event.

    :param event: The event to parse.
    :returns: The part ID and tagged flag if the event is a stream-part event,
              otherwise None.
    """

    if event.type != Gst.EventType.CUSTOM_DOWNSTREAM:
        return None

    struct: Gst.Structure = event.get_structure()
    if not struct.has_name(STREAM_PART_EVENT_NAME):
        return None

    parsed, part_id = struct.get_int(STREAM_PART_EVENT_PART_ID_PROPERTY)
    if not parsed:
        return None

    parsed, tagged = struct.get_boolean(STREAM_PART_EVENT_TAGGED_PROPERTY)
    if not parsed:
        return None

    return part_id, tagged
