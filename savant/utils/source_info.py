"""SourceInfo structure and SourceInfoRegistry singleton."""
from typing import List, Optional, Dict
from threading import Event
from dataclasses import dataclass
from savant.gstreamer import Gst
from savant.utils.singleton import SingletonMeta


@dataclass
class SourceInfo:
    """Source info."""

    source_id: str
    pad_idx: Optional[int]
    before_muxer: List[Gst.Element]
    after_demuxer: List[Gst.Element]
    lock: Event


class SourceInfoRegistry(metaclass=SingletonMeta):
    """Source info regsitry, provides access to
    ``{source_id : SourceInfo}`` map and ``{pad_index : source_id}`` map.
    """

    def __init__(self):
        self._sources: Dict[str, SourceInfo] = {}
        self._source_id_by_index: Dict[int, str] = {}

    def set_src_info(self, source_id: str, source_info: SourceInfo) -> None:
        """Store a source info structure with source id key.

        :param source_id: String to be used as a key for storing SourceInfo structure.
        :param source_info: SourceInfo structure to be stored.
        """
        self._sources[source_id] = source_info

    def set_src_id(self, pad_idx: int, source_id: str) -> None:
        """Store a source id string with pad index key.

        :param pad_idx: Int to be used as a key for storing source id string.
        :param source_id: string to be stored.
        """
        self._source_id_by_index[pad_idx] = source_id

    def free_src_id(self, source_id: str) -> None:
        """Delete a given source id key from internal map.

        :param source_id: string key to be removed from map.
        """
        del self._sources[source_id]

    def free_pad_idx(self, pad_idx: int) -> None:
        """Delete a given pad index key from internal map.

        :param pad_idx: int key to be removed from map.
        """
        del self._source_id_by_index[pad_idx]

    def get_src_id(self, pad_idx: int) -> str:
        """Retrieve string value associated with given pad index.

        :param pad_idx: Key used to retrieve source id value.
        :return: Source id value.
        """
        return self._source_id_by_index[pad_idx]

    def get_src_info(self, source_id: str) -> SourceInfo:
        """Retrieve SourceInfo value associated with given source id.

        :param source_id: Key used to retrieve SourceInfo value.
        :return: SourceInfo value.
        """
        return self._sources[source_id]
