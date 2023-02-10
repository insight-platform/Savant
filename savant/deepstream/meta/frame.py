"""Wrapper of deepstream frame meta information."""
from typing import Iterator, Any, Dict
import pyds

from savant.gstreamer.metadata import METADATA_STORAGE
from savant.meta.errors import MetaValueError
from savant.deepstream.meta.iterators import NvDsObjectMetaIterator
from savant.deepstream.meta.object import _NvDsObjectMetaImpl
from savant.meta.object import ObjectMeta
from savant.utils.source_info import SourceInfoRegistry
from pygstsavantframemeta import nvds_frame_meta_get_nvds_savant_frame_meta


class NvDsFrameMeta:
    """Wrapper of deepstream frame meta information.

    :param frame_meta: Deepstream python bindings frame meta.
    """

    def __init__(
        self,
        frame_meta: pyds.NvDsFrameMeta,
    ):
        super().__init__()
        self.batch_meta = frame_meta.base_meta.batch_meta
        self.frame_meta = frame_meta
        self.savant_frame_meta = nvds_frame_meta_get_nvds_savant_frame_meta(frame_meta)

    @property
    def source_id(self) -> str:
        """Source id for the frame in the batch."""
        return SourceInfoRegistry().get_id_by_pad_index(self.frame_meta.pad_index)

    @property
    def frame_num(self) -> int:
        """Current frame number of the source."""
        return self.frame_meta.frame_num

    @property
    def objects(self) -> Iterator[ObjectMeta]:
        """Returns an iterator over object metas in current frame.

        :return: Iterator over object metas.
        """
        return NvDsObjectMetaIterator(self.frame_meta)

    @property
    def objects_number(self) -> int:
        """Returns number of objects in frame meta.

        :return: Objects number.
        """
        return self.frame_meta.num_obj_meta

    @property
    def tags(self) -> Dict[str, Any]:
        """Returns tags of frame. These tags are part of the meta information about
        the frame that comes with the frames in the module.

        :return: Dictionary with tags
        """
        source_meta = METADATA_STORAGE[self.source_id]
        return source_meta.by_idx.get(self.savant_frame_meta.idx).tags

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
