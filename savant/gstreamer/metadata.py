"""Gstreamer metadata."""
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Union

logger = logging.getLogger(__name__)

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

    by_idx: Dict[int, GstFrameMeta] = field(default_factory=dict)
    """Frame IDX -> frame metadata."""
    by_pts: Dict[int, GstFrameMeta] = field(default_factory=dict)
    """Frame PTS -> frame metadata. For cases when frame doesn't have IDX."""


# metadata storage, workaround
# a way to pass metadata from src to output converter
# {source_id: {pts: metadata}}
METADATA_STORAGE: Dict[str, GstSourceMeta] = defaultdict(GstSourceMeta)


def metadata_add_frame_meta(
    source_id: str,
    frame_idx: Optional[int],
    frame_pts: int,
    frame_meta: GstFrameMeta,
) -> None:
    """Add metadata to frame.

    :param source_id: Source identifier.
    :param frame_idx: Frame index.
    :param frame_pts: Frame presentation timestamp.
    :param frame_meta: Frame metadata storage.
    """
    logger.debug(
        'Add metadata for frame of source %s with IDX %s and PTS %s.',
        source_id,
        frame_idx,
        frame_pts,
    )
    source_meta = METADATA_STORAGE[source_id]
    if frame_idx is not None:
        source_meta.by_idx[frame_idx] = frame_meta
    else:
        source_meta.by_pts[frame_pts] = frame_meta


def metadata_get_frame_meta(
    source_id: str,
    frame_idx: Optional[int],
    frame_pts: int,
) -> GstFrameMeta:
    """Get metadata from frame.

    :param source_id: Source identifier.
    :param frame_idx: Frame index.
    :param frame_pts: Frame presentation timestamp.
    :return: Metadata storage for the given frame.
    """
    logger.debug(
        'Get metadata for frame of source %s with IDX %s and PTS %s.',
        source_id,
        frame_idx,
        frame_pts,
    )
    source_meta = METADATA_STORAGE[source_id]
    if frame_idx is not None:
        frame_meta = source_meta.by_idx.get(frame_idx)
    else:
        frame_meta = source_meta.by_pts.get(frame_pts)
    if frame_meta is None:
        frame_meta = GstFrameMeta(source_id, frame_pts)
    return frame_meta


def metadata_pop_frame_meta(
    source_id: str,
    frame_idx: Optional[int],
    frame_pts: int,
) -> GstFrameMeta:
    """Get metadata from frame and remove it.

    :param source_id: Source identifier.
    :param frame_idx: Frame index.
    :param frame_pts: Frame presentation timestamp.
    :return: Metadata storage for the given frame.
    """
    logger.debug(
        'Pop metadata for frame of source %s with IDX %s and PTS %s.',
        source_id,
        frame_idx,
        frame_pts,
    )
    source_meta = METADATA_STORAGE[source_id]
    if frame_idx is not None:
        frame_meta = source_meta.by_idx.pop(frame_idx, None)
    else:
        frame_meta = source_meta.by_pts.pop(frame_pts, None)
    if frame_meta is None:
        frame_meta = GstFrameMeta(source_id, frame_pts)
    return frame_meta


def metadata_remove_frame_meta(
    source_id: str,
    frame_idx: Optional[int],
    frame_pts: int,
) -> None:
    """Remove metadata from frame.

    :param source_id: Source identifier.
    :param frame_idx: Frame index.
    :param frame_pts: Frame presentation timestamp.
    """
    logger.debug(
        'Remove metadata for frame of source %s with IDX %s and PTS %s.',
        source_id,
        frame_idx,
        frame_pts,
    )
    source_meta = METADATA_STORAGE[source_id]
    if frame_idx is not None:
        if frame_idx in source_meta.by_idx:
            del source_meta.by_idx[frame_idx]
    else:
        if frame_pts in source_meta.by_pts:
            del source_meta.by_pts[frame_pts]
