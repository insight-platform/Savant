"""Wrapper of deepstream frame meta information."""
from typing import Iterator
import pyds
from savant.meta.errors import MetaValueError
from savant.deepstream.meta.iterators import NvDsObjectMetaIterator
from savant.deepstream.meta.object import _NvDsObjectMetaImpl
from savant.meta.object import ObjectMeta
from savant.utils.source_info import SourceInfoRegistry


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

    @property
    def source_id(self) -> str:
        """Source id for the frame in the batch."""
        return SourceInfoRegistry().get_src_id(self.frame_meta.pad_index)

    @property
    def is_initial(self) -> bool:
        """Flag indicating whether this frame is the initial one for a video stream
        or a continuation of an established video stream.
        """
        return self.frame_meta.frame_num == 0

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
