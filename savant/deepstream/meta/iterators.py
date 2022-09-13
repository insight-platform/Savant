"""Iterator over deepstream object metadata."""
import pyds
from savant.deepstream.meta.object import _NvDsObjectMetaImpl
from savant.meta.object import ObjectMeta


class NvDsObjectMetaIterator:
    """Iterator over deepstream object metadata."""

    def __init__(self, frame_meta: pyds.NvDsFrameMeta):
        self.item = frame_meta.obj_meta_list
        self.cast_data = pyds.NvDsObjectMeta.cast
        self.frame_meta = frame_meta

    def __iter__(self):
        return self

    def __next__(self) -> ObjectMeta:
        item = self.item
        if item is None:
            raise StopIteration
        self.item = self.item.next
        return ObjectMeta._from_be_object_meta(
            _NvDsObjectMetaImpl.from_nv_ds_object_meta(
                self.cast_data(item.data), frame_meta=self.frame_meta
            )
        )
