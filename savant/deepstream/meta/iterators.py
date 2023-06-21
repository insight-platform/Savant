"""Iterator over deepstream object metadata."""
from typing import Iterator, ContextManager
from contextlib import contextmanager
import pyds
from savant.deepstream.meta.object import _NvDsObjectMetaImpl
from savant.meta.object import ObjectMeta

import logging

logger = logging.getLogger(__name__)

def finalize_obj_meta(obj_meta: ObjectMeta) -> None:
    logger.info(f'calling sync bbox obj_meta: {id(obj_meta)}')
    obj_meta.sync_bbox()


@contextmanager
def obj_meta_manager(
    obj_meta: pyds.NvDsObjectMeta, frame_meta: pyds.NvDsFrameMeta
) -> ContextManager[ObjectMeta]:
    obj_meta = ObjectMeta._from_be_object_meta(
        _NvDsObjectMetaImpl.from_nv_ds_object_meta(obj_meta, frame_meta)
    )
    try:
        logger.info(f'yielding obj_meta manager: {id(obj_meta)}')
        yield obj_meta
    finally:
        logger.info(f'finalizing obj_meta: {id(obj_meta)}')
        finalize_obj_meta(obj_meta)


def nvds_obj_meta_generator(frame_meta: pyds.NvDsFrameMeta) -> Iterator[ObjectMeta]:
    item = frame_meta.obj_meta_list
    while item is not None:
        with obj_meta_manager(
            pyds.NvDsObjectMeta.cast(item.data), frame_meta
        ) as obj_meta:
            logger.info(f'yielding obj_meta generator: {id(obj_meta)}')
            yield obj_meta
        item = item.next
