"""Wrapper of deepstream frame meta information."""
from typing import Iterator, Optional
import pyds
from savant_rs.primitives.geometry import BBox
from savant.gstreamer.metadata import (
    get_source_frame_meta,
    SourceFrameMeta,
    OnlyExtendedDict,
)
from savant.meta.errors import MetaValueError
from savant.deepstream.meta.iterators import nvds_obj_meta_generator
from savant.deepstream.meta.object import _NvDsObjectMetaImpl

from savant.meta.object import ObjectMeta
from savant.utils.source_info import SourceInfoRegistry
from pygstsavantframemeta import nvds_frame_meta_get_nvds_savant_frame_meta

import logging

logger = logging.getLogger(__name__)

class NvDsFrameMeta:
    """Wrapper of deepstream frame meta information.

    :param frame_meta: Deepstream python bindings frame meta.
    """

    def __init__(
        self,
        frame_meta: pyds.NvDsFrameMeta,
    ):
        super().__init__()
        self.batch_meta: pyds.NvDsBatchMeta = frame_meta.base_meta.batch_meta
        self.frame_meta: pyds.NvDsFrameMeta = frame_meta
        self._source_frame_meta: Optional[SourceFrameMeta] = None
        self._primary_obj: Optional[ObjectMeta] = None

    @property
    def source_id(self) -> str:
        """Source id for the frame in the batch."""
        return SourceInfoRegistry().get_id_by_pad_index(self.frame_meta.pad_index)

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
        logger.info(f'calling nvds_obj_meta_generator')
        return nvds_obj_meta_generator(self.frame_meta)

    @property
    def roi(self) -> BBox:
        logger.info(f"Getting ROI")
        if not self._primary_obj:
            for obj_meta in self.objects:
                if obj_meta.is_primary:
                    self._primary_obj = obj_meta
                    break
        return self._primary_obj.bbox

    @roi.setter
    def roi(self, value: BBox):
        logger.info(f"Setting ROI to {value}")
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

    @property
    def tags(self) -> OnlyExtendedDict:
        """Returns tags of frame. These tags are part of the meta information about
        the frame that comes with the frames in the module.

        :return: Dictionary with tags
        """
        if self._source_frame_meta is None:
            self._set_source_frame_meta()
        return self._source_frame_meta.tags

    def _set_source_frame_meta(self):
        """Set the source frame metadata.

        :return: None
        """
        savant_frame_meta = nvds_frame_meta_get_nvds_savant_frame_meta(self.frame_meta)
        self._source_frame_meta = get_source_frame_meta(
            source_id=self.source_id,
            frame_idx=savant_frame_meta.idx if savant_frame_meta else None,
            frame_pts=self.frame_meta.buf_pts,
        )

    @property
    def pts(self) -> int:
        """Get the presentation time stamp (PTS) of the current frame.

        :return: The PTS of the current frame, if available; None otherwise.
        """
        if self._source_frame_meta is None:
            self._set_source_frame_meta()
        return self._source_frame_meta.pts

    @property
    def duration(self) -> Optional[int]:
        """Get the duration of the current frame.

        :returns: The duration of the current frame, if available; None otherwise.
        """
        if self._source_frame_meta is None:
            self._set_source_frame_meta()
        return self._source_frame_meta.duration

    @property
    def framerate(self) -> str:
        """Get the framerate of the current frame.

        returns: The framerate of the current frame as a string.
        """
        if self._source_frame_meta is None:
            self._set_source_frame_meta()
        return self._source_frame_meta.framerate

    def add_obj_meta(self, object_meta: ObjectMeta):
        """Add an object meta to frame meta.

        :param object_meta: Object meta to add.
        """
        if isinstance(object_meta, ObjectMeta):
            if object_meta.object_meta_impl and isinstance(
                object_meta.object_meta_impl, pyds.NvDsObjectMeta
            ):
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
            if object_meta.object_meta_impl:
                pyds.nvds_remove_obj_meta_from_frame(
                    self.frame_meta, object_meta.object_meta_impl.ds_object_meta
                )
        else:
            raise MetaValueError(
                f"{self.__class__.__name__} doesn't "
                f'support removing object meta `of {type(object_meta)}` type'
            )
