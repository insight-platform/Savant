"""SourceInfo structure and SourceInfoRegistry singleton."""

from dataclasses import dataclass
from threading import Event
from typing import Dict, List, Optional

from savant.config.schema import FramePadding
from savant.gstreamer import Gst
from savant.utils.singleton import SingletonMeta


@dataclass
class Resolution:
    width: int
    height: int


@dataclass
class SourceShape:
    """Pipeline processing frame shape"""

    width: int
    """Pipeline processing frame width"""

    height: int
    """Pipeline processing frame height"""

    padding: Optional[FramePadding] = None
    """Add paddings to the frame before processing"""


@dataclass
class SourceInfo:
    """Source info."""

    source_id: str
    pad_idx: Optional[int]
    src_resolution: Optional[Resolution]
    shape: Optional[SourceShape]
    before_muxer: List[Gst.Element]
    source_output_elements: List[Gst.Element]
    after_demuxer: List[Gst.Element]
    lock: Event

    @property
    def processing_width(self):
        """Width of the frame that will be processed by the pipeline."""

        return self.shape.width

    @property
    def processing_height(self):
        """Height of the frame that will be processed by the pipeline."""

        return self.shape.height

    @property
    def total_width(self) -> int:
        """Total frame width including paddings."""

        if self.shape.padding is not None:
            return self.processing_width + self.shape.padding.width
        return self.processing_width

    @property
    def total_height(self) -> int:
        """Total frame height including paddings."""

        if self.shape.padding is not None:
            return self.processing_height + self.shape.padding.height
        return self.processing_height

    @property
    def output_width(self) -> int:
        """Width of the output frame. Includes paddings if they are set to keep."""

        if self.shape.padding is not None and self.shape.padding.keep:
            return self.total_width
        return self.processing_width

    @property
    def output_height(self) -> int:
        """Height of the output frame. Includes paddings if they are set to keep."""

        if self.shape.padding is not None and self.shape.padding.keep:
            return self.total_height
        return self.processing_height

    @property
    def padding(self) -> Optional[FramePadding]:
        """Frame padding."""
        return self.shape.padding


class SourceInfoRegistry(metaclass=SingletonMeta):
    """Source info registry, provides access to
    ``{source_id : SourceInfo}`` map and ``{pad_index : source_id}`` map.
    """

    def __init__(self):
        self._sources: Dict[str, SourceInfo] = {}
        self._source_id_by_index: Dict[int, str] = {}

    def init_source(self, source_id: str, shape: Optional[SourceShape]) -> SourceInfo:
        source_info = SourceInfo(
            source_id=source_id,
            pad_idx=None,
            src_resolution=None,
            shape=shape,
            before_muxer=[],
            source_output_elements=[],
            after_demuxer=[],
            lock=Event(),
        )
        self._sources[source_id] = source_info
        return source_info

    def get_source(self, source_id: str) -> SourceInfo:
        """Retrieve SourceInfo value associated with given source id.

        :param source_id: Key used to retrieve SourceInfo value.
        :return: SourceInfo value.
        """
        return self._sources[source_id]

    def update_source(self, source_info: SourceInfo) -> None:
        """Update internal maps for source_id and pad index
        of a given SourceInfo structure.

        :param source_info: SourceInfo structure to be stored.
        """
        self._sources[source_info.source_id] = source_info
        self._source_id_by_index[source_info.pad_idx] = source_info.source_id

    def remove_source(self, source_info: SourceInfo) -> None:
        """Delete a given source info entries from internal map.

        :param source_info: SourceInfo to be removed from map.
        """
        del self._sources[source_info.source_id]
        if source_info.pad_idx is not None:
            del self._source_id_by_index[source_info.pad_idx]

    def get_id_by_pad_index(self, pad_idx: int) -> str:
        """Retrieve string value associated with given pad index.

        :param pad_idx: Key used to retrieve source id value.
        :return: Source id value.
        """
        return self._source_id_by_index[pad_idx]

    def has_sources(self):
        """Check if there are any sources registered."""
        return bool(self._sources)
