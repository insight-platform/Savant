"""Wrapper of deepstream frame meta information."""
from contextlib import AbstractContextManager
from typing import Dict, Iterator, Optional, Union

import pyds
from savant_rs.primitives import Attribute, VideoFrame
from savant_rs.primitives.geometry import BBox
from savant_rs.utils import TelemetrySpan

from savant.api.builder import build_attribute_value
from savant.api.constants import DEFAULT_NAMESPACE
from savant.api.parser import parse_attribute_value
from savant.deepstream.meta.object import _NvDsObjectMetaImpl
from savant.meta.errors import MetaValueError
from savant.meta.object import ObjectMeta
from savant.utils.logging import LoggerMixin


def nvds_obj_meta_generator(
    frame_meta: pyds.NvDsFrameMeta, obj_cache: Dict[int, ObjectMeta]
) -> Iterator[ObjectMeta]:
    item = frame_meta.obj_meta_list
    while item is not None:
        nvds_obj_meta = pyds.NvDsObjectMeta.cast(item.data)
        obj_meta = ObjectMeta._from_be_object_meta(
            _NvDsObjectMetaImpl.from_nv_ds_object_meta(nvds_obj_meta, frame_meta)
        )

        try:
            ret_item = obj_cache[obj_meta.uid]
        except KeyError:
            ret_item = obj_meta
            obj_cache[obj_meta.uid] = obj_meta

        yield ret_item
        item = item.next


class NvDsFrameMeta(AbstractContextManager, LoggerMixin):
    """Wrapper of deepstream frame meta information.

    :param frame_meta: Deepstream python bindings frame meta.
    :param video_frame: Video frame meta.
    :param telemetry_span: The telemetry span associated with the frame.
    """

    def __init__(
        self,
        frame_meta: pyds.NvDsFrameMeta,
        video_frame: VideoFrame,
        telemetry_span: TelemetrySpan,
    ):
        super().__init__()
        self.batch_meta: pyds.NvDsBatchMeta = frame_meta.base_meta.batch_meta
        self.frame_meta: pyds.NvDsFrameMeta = frame_meta
        self._video_frame: VideoFrame = video_frame
        self._telemetry_span: TelemetrySpan = telemetry_span
        self._primary_obj: Optional[ObjectMeta] = None
        self._objects = {}

    def __exit__(self, *exc_details):
        self.logger.debug(
            'Syncing bbox for all objects in frame with PTS %s.',
            self.frame_meta.buf_pts,
        )
        for obj in self._objects.values():
            obj.sync_bbox()
        self._objects.clear()
        return super().__exit__(*exc_details)

    @property
    def source_id(self) -> str:
        """Source id for the frame in the batch."""
        return self._video_frame.source_id

    @property
    def frame_num(self) -> int:
        """Current frame number of the source."""
        return self.frame_meta.frame_num

    @property
    def batch_id(self) -> int:
        """Location of the frame in the batch."""
        return self.frame_meta.batch_id

    @property
    def objects(self) -> Iterator[ObjectMeta]:
        """Returns an iterator over object metas in current frame.

        :return: Iterator over object metas.
        """
        return nvds_obj_meta_generator(self.frame_meta, self._objects)

    @property
    def roi(self) -> BBox:
        if not self._primary_obj:
            for obj_meta in self.objects:
                if obj_meta.is_primary:
                    self._primary_obj = obj_meta
                    break
        return self._primary_obj.bbox

    @roi.setter
    def roi(self, value: BBox):
        self.roi.xc = value.xc
        self.roi.yc = value.yc
        self.roi.width = value.width
        self.roi.height = value.height

    @property
    def objects_number(self) -> int:
        """Returns number of objects in frame meta.

        :return: Objects number.
        """
        return self.frame_meta.num_obj_meta

    def get_tag(self, name: str) -> Optional[Union[bool, int, float, str]]:
        """Get tag of frame. These tags are part of the meta information about
        the frame that comes with the frames in the module.

        :return: Tag value
        """

        attr = self._video_frame.get_attribute(DEFAULT_NAMESPACE, name)
        if attr is not None:
            return parse_attribute_value(attr.values[0])

    def set_tag(self, name: str, value: Union[bool, int, float, str]):
        """Set tag to frame. These tags are part of the meta information about
        the frame that comes with the frames in the module.

        :param name: Tag name
        :param value: Tag value
        """

        self._video_frame.set_attribute(
            Attribute(
                namespace=DEFAULT_NAMESPACE,
                name=name,
                values=[build_attribute_value(value)],
            )
        )

    @property
    def pts(self) -> int:
        """Get the presentation time stamp (PTS) of the current frame.

        :return: The PTS of the current frame, if available; None otherwise.
        """
        return self._video_frame.pts

    @property
    def duration(self) -> Optional[int]:
        """Get the duration of the current frame.

        :returns: The duration of the current frame, if available; None otherwise.
        """
        return self._video_frame.duration

    @property
    def framerate(self) -> str:
        """Get the framerate of the current frame.

        returns: The framerate of the current frame as a string.
        """
        return self._video_frame.framerate

    def add_obj_meta(self, object_meta: ObjectMeta):
        """Add an object meta to frame meta.

        :param object_meta: Object meta to add.
        """
        if isinstance(object_meta, ObjectMeta):
            if object_meta.object_meta_impl and isinstance(
                object_meta.object_meta_impl, pyds.NvDsObjectMeta
            ):
                return

            if object_meta.uid is not None and object_meta.uid in self._objects:
                return

            ds_object_meta = _NvDsObjectMetaImpl(
                frame_meta=self,
                element_name=object_meta.element_name,
                label=object_meta.label,
                bbox=object_meta.bbox,
                confidence=object_meta.confidence,
                track_id=object_meta.track_id,
                parent=object_meta.parent,
            )
            object_meta.object_meta_impl = ds_object_meta
            self._objects[object_meta.uid] = object_meta
        else:
            raise MetaValueError(
                f"{self.__class__.__name__} doesn't "
                f'support adding object meta `of {type(object_meta)}` type'
            )

    def remove_obj_meta(self, object_meta: ObjectMeta):
        """Remove an object meta from frame meta.

        :param object_meta: Object meta to remove.
        """
        if isinstance(object_meta, ObjectMeta):
            if object_meta.uid in self._objects:
                del self._objects[object_meta.uid]
            if object_meta.object_meta_impl:
                pyds.nvds_remove_obj_meta_from_frame(
                    self.frame_meta, object_meta.object_meta_impl.ds_object_meta
                )
                object_meta.object_meta_impl = None
        else:
            raise MetaValueError(
                f"{self.__class__.__name__} doesn't "
                f'support removing object meta `of {type(object_meta)}` type'
            )

    @property
    def video_frame(self) -> VideoFrame:
        """Get the video frame associated with the frame meta."""
        return self._video_frame

    @property
    def telemetry_span(self) -> TelemetrySpan:
        """Get the telemetry span associated with the frame.

        Example:

        .. code-block:: python

            with frame_meta.telemetry_span.nested_span("process-frame"):
                # do something
        """
        return self._telemetry_span
