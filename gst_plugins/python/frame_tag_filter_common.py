from typing import Optional, Tuple

from savant.gstreamer import Gst

STREAM_PART_EVENT_NAME = 'stream-part'
STREAM_PART_EVENT_TAGGED_PROPERTY = 'tagged'
STREAM_PART_EVENT_PART_ID_PROPERTY = 'part-id'


def build_stream_part_event(part_id: int, tagged: bool):
    structure: Gst.Structure = Gst.Structure.new_empty(STREAM_PART_EVENT_NAME)
    structure.set_value(STREAM_PART_EVENT_PART_ID_PROPERTY, part_id)
    structure.set_value(STREAM_PART_EVENT_TAGGED_PROPERTY, tagged)
    return Gst.Event.new_custom(Gst.EventType.CUSTOM_DOWNSTREAM, structure)


def parse_stream_part_event(event: Gst.Event) -> Optional[Tuple[int, bool]]:
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
