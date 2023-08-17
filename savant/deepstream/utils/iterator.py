"""DeepStream iterators."""
from typing import Any, Callable, Optional, Union

import pyds


class NvDsMetaIterator:
    """NvDs metadata iterator.

    :param item: An object with `data` (optional) and `next` fields
    :param cast_data: A function to cast data
    """

    def __init__(self, item: object, cast_data: Optional[Callable] = None):
        self.item = item
        self.cast_data = cast_data

    def __iter__(self):
        return self

    def __next__(self):
        item = self.item
        if item is None:
            raise StopIteration
        self.item = self.item.next
        if self.cast_data:
            return self.cast_data(item.data)
        return item


def nvds_frame_meta_iterator(batch_meta: pyds.NvDsBatchMeta) -> NvDsMetaIterator:
    """NvDsFrameMeta iterator.

    :param batch_meta: NvDs batch metadata structure that will be iterated on.
    :return: NvDsFrameMeta iterator.
    """
    return NvDsMetaIterator(batch_meta.frame_meta_list, pyds.NvDsFrameMeta.cast)


def nvds_batch_user_meta_iterator(batch_meta: pyds.NvDsBatchMeta) -> NvDsMetaIterator:
    """NvDsUserMeta iterator.

    :param batch_meta: NvDs batch metadata structure that will be iterated on.
    :return: NvDsUserMeta iterator.
    """
    return NvDsMetaIterator(batch_meta.batch_user_meta_list, pyds.NvDsUserMeta.cast)


def nvds_frame_user_meta_iterator(frame_meta: pyds.NvDsFrameMeta) -> NvDsMetaIterator:
    """NvDsUserMeta iterator.

    :param frame_meta: NvDs frame metadata structure that will be iterated on.
    :return: NvDsUserMeta iterator.
    """
    return NvDsMetaIterator(frame_meta.frame_user_meta_list, pyds.NvDsUserMeta.cast)


def nvds_obj_user_meta_iterator(obj_meta: pyds.NvDsObjectMeta) -> NvDsMetaIterator:
    """NvDsUserMeta iterator.

    :param frame_meta: NvDs object metadata structure that will be iterated on.
    :return: NvDsUserMeta iterator.
    """
    return NvDsMetaIterator(obj_meta.obj_user_meta_list, pyds.NvDsUserMeta.cast)


def nvds_obj_meta_iterator(frame_meta: pyds.NvDsFrameMeta) -> NvDsMetaIterator:
    """NvDsObjectMeta iterator.

    :param frame_meta: NvDs frame metadata structure that will be iterated on.
    :return: NvDsObjectMeta iterator.
    """
    return NvDsMetaIterator(frame_meta.obj_meta_list, pyds.NvDsObjectMeta.cast)


def nvds_clf_meta_iterator(obj_meta: pyds.NvDsObjectMeta) -> NvDsMetaIterator:
    """NvDsClassifierMeta iterator.

    :param obj_meta: NvDs object metadata structure that will be iterated on.
    :return: NvDsClassifierMeta iterator.
    """
    return NvDsMetaIterator(obj_meta.classifier_meta_list, pyds.NvDsClassifierMeta.cast)


def nvds_label_info_iterator(clf_meta: pyds.NvDsClassifierMeta) -> NvDsMetaIterator:
    """NvDsLabelInfo(result_class_id, result_label, result_prob etc.) iterator.

    :param clf_meta: NvDs classifier metadata structure that will be iterated on.
    :return: NvDsLabelInfo iterator
    """
    return NvDsMetaIterator(clf_meta.label_info_list, pyds.NvDsLabelInfo.cast)


def nvds_tensor_output_iterator(
    meta: Union[pyds.NvDsFrameMeta, pyds.NvDsObjectMeta], gie_uid: Optional[int] = None
) -> NvDsMetaIterator:
    """NvDsUserMeta + NvDsInferTensorMeta iterator.

    :param meta: NvDs frame or object metadata structure that will be iterated on.
    :param gie_uid: inference engine id, `gie-unique-id`.
    :return: NvDsInferTensorMeta iterator.
    """

    def cast_data(data: Any):
        """Double cast and filter."""
        user_meta = pyds.NvDsUserMeta.cast(data)
        if user_meta.base_meta.meta_type == pyds.NVDSINFER_TENSOR_OUTPUT_META:
            tensor_meta = pyds.NvDsInferTensorMeta.cast(user_meta.user_meta_data)
            if gie_uid is None or gie_uid == tensor_meta.unique_id:
                return tensor_meta
        return None

    user_meta_list = (
        meta.frame_user_meta_list
        if isinstance(meta, pyds.NvDsFrameMeta)
        else meta.obj_user_meta_list
    )
    for tensor_output in NvDsMetaIterator(user_meta_list, cast_data=cast_data):
        if tensor_output:
            yield tensor_output
