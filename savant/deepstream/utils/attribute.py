"""DeepStream object attribute utils."""

import logging
from collections import OrderedDict
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pyds
from pygstsavantframemeta import nvds_frame_meta_get_nvds_savant_frame_meta

from savant.meta.attribute import AttributeMeta

from .object import nvds_get_obj_uid

logger = logging.getLogger(__name__)

# Frames kept before the oldest is dropped. Only frames that never reach the
# output probe age out, so the cap is a backstop, not a working limit.
MAX_TRACKED_FRAMES = 1000

FrameKey = Tuple
ObjAttrs = Dict[Tuple[str, str], List[AttributeMeta]]


@dataclass
class _FrameAttrs:
    """Attributes of all objects of a single frame."""

    pad_idx: int
    objects: Dict[int, ObjAttrs] = field(default_factory=dict)


class ObjAttrStorage:
    """Object attributes that DS object meta cannot hold, keyed by frame and
    object uid.

    Attributes live from the moment an element attaches them until the frame is
    converted for output, so the frame is the outer key: the output probe drops a
    whole frame at once and objects removed from the frame meta in between (e.g.
    by the tracker, in C) cannot leak.

    Writers run on the muxer, nvinfer and pyfunc streaming threads while the
    purge runs on the demuxer and EOS threads, so every method takes the lock.
    """

    def __init__(self, max_frames: int = MAX_TRACKED_FRAMES):
        self._frames: 'OrderedDict[FrameKey, _FrameAttrs]' = OrderedDict()
        self._lock = Lock()
        self._max_frames = max_frames
        self._evictions = 0
        self._next_warning = 1

    def add(  # pylint: disable=too-many-arguments
        self,
        key: FrameKey,
        pad_idx: int,
        uid: int,
        element_name: str,
        attr_name: str,
        attr: AttributeMeta,
        replace: bool,
    ):
        with self._lock:
            frame = self._frames.get(key)
            if frame is None:
                frame = self._frames[key] = _FrameAttrs(pad_idx=pad_idx)
            attrs = frame.objects.setdefault(uid, {})
            if replace or (element_name, attr_name) not in attrs:
                attrs[(element_name, attr_name)] = []
            attrs[(element_name, attr_name)].append(attr)
            evicted = self._evict()
        self._warn_evicted(evicted)

    def replace(  # pylint: disable=too-many-arguments
        self,
        key: FrameKey,
        pad_idx: int,
        uid: int,
        element_name: str,
        attr_name: str,
        value: List[AttributeMeta],
    ):
        with self._lock:
            frame = self._frames.get(key)
            if frame is None:
                frame = self._frames[key] = _FrameAttrs(pad_idx=pad_idx)
            frame.objects.setdefault(uid, {})[(element_name, attr_name)] = value
            evicted = self._evict()
        self._warn_evicted(evicted)

    def get(
        self, key: FrameKey, uid: int, element_name: str, attr_name: str
    ) -> Optional[List[AttributeMeta]]:
        with self._lock:
            frame = self._frames.get(key)
            if frame is None:
                return None
            return frame.objects.get(uid, {}).get((element_name, attr_name))

    def get_all(self, key: FrameKey, uid: int) -> List[AttributeMeta]:
        with self._lock:
            frame = self._frames.get(key)
            if frame is None:
                return []
            attrs = frame.objects.get(uid)
            if not attrs:
                return []
            return [attr for attr_list in attrs.values() for attr in attr_list]

    def remove_attr(self, key: FrameKey, uid: int, element_name: str, attr_name: str):
        with self._lock:
            frame = self._frames.get(key)
            if frame is None:
                return
            attrs = frame.objects.get(uid)
            if attrs is None:
                return
            attrs.pop((element_name, attr_name), None)
            if not attrs:
                del frame.objects[uid]
            self._drop_empty(key, frame)

    def remove_obj(self, key: FrameKey, uid: int):
        with self._lock:
            frame = self._frames.get(key)
            if frame is None:
                return
            frame.objects.pop(uid, None)
            self._drop_empty(key, frame)

    def remove_frame(self, key: FrameKey):
        with self._lock:
            self._frames.pop(key, None)

    def remove_source(self, pad_idx: int):
        with self._lock:
            for key in [
                key for key, frame in self._frames.items() if frame.pad_idx == pad_idx
            ]:
                del self._frames[key]

    def __len__(self) -> int:
        """Number of objects held, over all frames."""
        with self._lock:
            return sum(len(frame.objects) for frame in self._frames.values())

    def values(self) -> List[ObjAttrs]:
        """Snapshot of the per-object attribute maps."""
        with self._lock:
            return [
                dict(attrs)
                for frame in self._frames.values()
                for attrs in frame.objects.values()
            ]

    def _drop_empty(self, key: FrameKey, frame: _FrameAttrs):
        """Caller holds the lock."""
        if not frame.objects:
            del self._frames[key]

    def _evict(self) -> Optional[int]:
        """Drop the oldest frames over the cap, returns the total to warn about.

        Caller holds the lock.
        """
        while len(self._frames) > self._max_frames:
            self._frames.popitem(last=False)
            self._evictions += 1
        if self._evictions < self._next_warning:
            return None
        self._next_warning = self._evictions * 2
        return self._evictions

    def _warn_evicted(self, evicted: Optional[int]):
        # A frame is evicted only if it never reached the output probe, i.e. a
        # purge is missing. Warn on the 1st, 2nd, 4th, ... occurrence.
        if evicted is not None:
            logger.warning(
                'Attribute storage is over %d frames, dropped the oldest '
                '(%d frames dropped so far).',
                self._max_frames,
                evicted,
            )


NVDS_OBJ_ATTR_STORAGE = ObjAttrStorage()


def _frame_key(frame_meta: pyds.NvDsFrameMeta) -> FrameKey:
    """Storage key of the given frame.

    A frame without savant frame meta is discarded by the pipeline, but elements
    may still attach attributes to its objects; keying it per frame keeps those
    purgeable by the output probe and subject to the frame cap.
    """
    savant_frame_meta = nvds_frame_meta_get_nvds_savant_frame_meta(frame_meta)
    if savant_frame_meta is not None:
        return (savant_frame_meta.idx,)
    return (None, frame_meta.pad_index, frame_meta.buf_pts)


def nvds_add_attr_meta_to_obj(  # pylint: disable=too-many-arguments
    frame_meta: pyds.NvDsFrameMeta,
    obj_meta: pyds.NvDsObjectMeta,
    element_name: str,
    name: str,
    value: Any,
    confidence: float = 1.0,
    replace: bool = False,
):
    """Adds attribute to the object.

    :param frame_meta: object parent frame.
    :param obj_meta: object metadata.
    :param element_name: element name that created this attribute.
    :param name: attribute name.
    :param value: attribute value.
    :param confidence: object confidence.
    :param replace: replace existing attribute.
    """
    NVDS_OBJ_ATTR_STORAGE.add(
        key=_frame_key(frame_meta),
        pad_idx=frame_meta.pad_index,
        uid=nvds_get_obj_uid(frame_meta, obj_meta),
        element_name=element_name,
        attr_name=name,
        attr=AttributeMeta(
            element_name=element_name, name=name, value=value, confidence=confidence
        ),
        replace=replace,
    )


def nvds_attr_meta_iterator(
    frame_meta: pyds.NvDsFrameMeta,
    obj_meta: pyds.NvDsObjectMeta,
) -> Iterable[AttributeMeta]:
    """AttributeMeta iterator(iterable).

    :param frame_meta: object parent frame.
    :param obj_meta: object metadata.
    :return: object attributes.
    """
    return NVDS_OBJ_ATTR_STORAGE.get_all(
        key=_frame_key(frame_meta),
        uid=nvds_get_obj_uid(frame_meta, obj_meta),
    )


def nvds_get_obj_attr_meta_list(
    frame_meta: pyds.NvDsFrameMeta,
    obj_meta: pyds.NvDsObjectMeta,
    element_name: str,
    attr_name: str,
) -> Optional[List[AttributeMeta]]:
    """Returns specified object attribute values (multi-label case).

    :param frame_meta: object parent frame.
    :param obj_meta: object metadata.
    :param element_name: element name that created this attribute.
    :param attr_name: attribute name.
    :return: List of AttributeMeta/None
    """
    return NVDS_OBJ_ATTR_STORAGE.get(
        key=_frame_key(frame_meta),
        uid=nvds_get_obj_uid(frame_meta, obj_meta),
        element_name=element_name,
        attr_name=attr_name,
    )


def nvds_get_obj_attr_meta(
    frame_meta: pyds.NvDsFrameMeta,
    obj_meta: pyds.NvDsObjectMeta,
    element_name: str,
    attr_name: str,
) -> Optional[AttributeMeta]:
    """Returns the first value (the first and only except in the case of a
    multi-label) for specified object attribute.

    :param frame_meta: object parent frame.
    :param obj_meta: object metadata.
    :param element_name: element name that created this attribute.
    :param attr_name: attribute name.
    :return: AttributeMeta/None
    """
    attrs = nvds_get_obj_attr_meta_list(frame_meta, obj_meta, element_name, attr_name)
    return attrs[0] if attrs else None


def nvds_replace_obj_attr_meta_list(
    frame_meta: pyds.NvDsFrameMeta,
    obj_meta: pyds.NvDsObjectMeta,
    element_name: str,
    attr_name: str,
    value: List[AttributeMeta],
):
    """Replaces specified object attribute values.

    :param frame_meta: object parent frame.
    :param obj_meta: object metadata.
    :param element_name: element name that created this attribute.
    :param attr_name: attribute name.
    :param value: new attribute value, list.
    """
    for attr in value:
        assert attr.element_name == element_name
        assert attr.name == attr_name
    NVDS_OBJ_ATTR_STORAGE.replace(
        key=_frame_key(frame_meta),
        pad_idx=frame_meta.pad_index,
        uid=nvds_get_obj_uid(frame_meta, obj_meta),
        element_name=element_name,
        attr_name=attr_name,
        value=value,
    )


def nvds_remove_obj_attr_meta_list(
    frame_meta: pyds.NvDsFrameMeta,
    obj_meta: pyds.NvDsObjectMeta,
    element_name: str,
    attr_name: str,
):
    """Removes specified object attribute values.

    :param frame_meta: object parent frame.
    :param obj_meta: object metadata.
    :param element_name: element name that created this attribute.
    :param attr_name: attribute name.
    """
    NVDS_OBJ_ATTR_STORAGE.remove_attr(
        key=_frame_key(frame_meta),
        uid=nvds_get_obj_uid(frame_meta, obj_meta),
        element_name=element_name,
        attr_name=attr_name,
    )


def nvds_remove_obj_attrs(
    frame_meta: pyds.NvDsFrameMeta,
    obj_meta: pyds.NvDsObjectMeta,
):
    """Removes object attributes (from NVDS_OBJ_ATTR_STORAGE).

    :param frame_meta: object parent frame.
    :param obj_meta: object metadata.
    """
    NVDS_OBJ_ATTR_STORAGE.remove_obj(
        key=_frame_key(frame_meta),
        uid=nvds_get_obj_uid(frame_meta, obj_meta),
    )


def nvds_remove_frame_attrs(frame_meta: pyds.NvDsFrameMeta):
    """Removes attributes of all objects of the frame.

    :param frame_meta: frame metadata.
    """
    NVDS_OBJ_ATTR_STORAGE.remove_frame(_frame_key(frame_meta))


def nvds_remove_source_attrs(pad_idx: int):
    """Removes attributes of all frames of the source.

    :param pad_idx: muxer pad index of the source.
    """
    NVDS_OBJ_ATTR_STORAGE.remove_source(pad_idx)
