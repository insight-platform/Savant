"""Gstreamer metadata."""
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Union

DEFAULT_FRAMERATE = '30/1'


@dataclass
class GstFrameMeta:
    """Gst frame metadata."""

    source_id: str
    pts: int
    duration: Optional[int] = None
    framerate: str = DEFAULT_FRAMERATE
    metadata: Dict[str, Any] = None
    tags: Dict[str, Union[str, bool, int, float]] = field(default_factory=dict)

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = dict(objects=[])


@dataclass
class GstSourceMeta:
    """Gst source metadata."""

    frames: Dict[int, GstFrameMeta] = field(default_factory=dict)
    """Frame PTS -> frame metadata."""


# metadata storage, workaround
# a way to pass metadata from src to output converter
# {source_id: {pts: metadata}}
METADATA_STORAGE: Dict[str, GstSourceMeta] = defaultdict(GstSourceMeta)


def metadata_add_frame_meta(
    source_id: str, frame_pts: int, frame_meta: GstFrameMeta
) -> None:
    """Add metadata to frame.

    :param source_id: Source identifier.
    :param frame_pts: Frame presentation timestamp.
    :param frame_meta: Frame metadata storage.
    """
    METADATA_STORAGE[source_id].frames[frame_pts] = frame_meta


def metadata_get_frame_meta(source_id: str, frame_pts: int) -> GstFrameMeta:
    """Get metadata from frame.

    :param source_id: Source identifier.
    :param frame_pts: Frame presentation timestamp.
    :return: Metadata storage for the given frame.
    """
    return METADATA_STORAGE[source_id].frames.get(
        frame_pts, GstFrameMeta(source_id, frame_pts)
    )


def metadata_pop_frame_meta(source_id: str, frame_pts: int) -> GstFrameMeta:
    """Get metadata from frame and remove it.

    :param source_id: Source identifier.
    :param frame_pts: Frame presentation timestamp.
    :return: Metadata storage for the given frame.
    """
    return METADATA_STORAGE[source_id].frames.pop(
        frame_pts, GstFrameMeta(source_id, frame_pts)
    )


def metadata_remove_frame_meta(source_id: str, frame_pts: int) -> None:
    """Remove metadata from frame.

    :param source_id: Source identifier.
    :param frame_pts: Frame presentation timestamp.
    """
    source_meta = METADATA_STORAGE[source_id]
    if frame_pts in source_meta.frames:
        del source_meta.frames[frame_pts]
