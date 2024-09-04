"""Deepstream-specific ObjectMeta interface implementation."""

import logging
from typing import Any, List, Optional, Union

import pyds
from savant_rs.primitives.geometry import BBox, RBBox
from savant_rs.utils.symbol_mapper import (
    build_model_object_key,
    get_model_name,
    get_object_id,
    parse_compound_key,
)

from savant.deepstream.meta.constants import MAX_LABEL_SIZE
from savant.deepstream.utils.attribute import (
    nvds_add_attr_meta_to_obj,
    nvds_get_obj_attr_meta,
    nvds_get_obj_attr_meta_list,
    nvds_remove_obj_attr_meta_list,
    nvds_replace_obj_attr_meta_list,
)
from savant.deepstream.utils.object import (
    nvds_get_obj_bbox,
    nvds_get_obj_draw_label,
    nvds_get_obj_uid,
    nvds_init_obj_draw_label,
    nvds_is_empty_object_meta,
    nvds_set_obj_bbox,
    nvds_set_obj_draw_label,
    nvds_set_obj_uid,
    nvds_upd_obj_bbox,
)
from savant.meta.attribute import AttributeMeta
from savant.meta.constants import DEFAULT_CONFIDENCE, UNTRACKED_OBJECT_ID
from savant.meta.errors import MetaValueError
from savant.meta.object import BaseObjectMetaImpl, ObjectMeta
from savant.utils.logging import LoggerMixin


class _NvDsObjectMetaImpl(BaseObjectMetaImpl, LoggerMixin):
    """The class implements the ObjectMeta interface for object meta from the
    Deepstream framework.

    :param frame_meta: Metadata for a frame to which object will be added.
    :param element_name: The unique identifier of the component by which
        the object was created.
    :param label: Class label of the object.
    :param bbox: Bounding box of the object.
    :param confidence: Confidence of the object from detector.
    :param track_id: Unique ID for tracking the object.
        Default value is :py:const:`~savant.meta.constants.UNTRACKED_OBJECT_ID`.
        It indicates that the object has not been tracked.
    :param parent: The parent object metadata.
    """

    def __init__(
        self,
        frame_meta: 'NvDsFrameMeta',
        element_name: str,
        label: str,
        bbox: Union[BBox, RBBox],
        confidence: Optional[float] = DEFAULT_CONFIDENCE,
        track_id: int = UNTRACKED_OBJECT_ID,
        parent: Optional['ObjectMeta'] = None,
        draw_label: Optional[str] = None,
    ):
        super().__init__()
        element_uid, class_id = get_object_id(element_name, label)
        self._frame_meta = frame_meta

        self.ds_object_meta: pyds.NvDsObjectMeta = pyds.nvds_acquire_obj_meta_from_pool(
            frame_meta.batch_meta
        )
        self.ds_object_meta.class_id = class_id
        self.ds_object_meta.unique_component_id = element_uid
        self.label = label  # MAX_LABEL_SIZE
        if draw_label is not None:
            self.draw_label = draw_label
        self.track_id = track_id
        self.parent = parent

        pyds.nvds_add_obj_meta_to_frame(
            self._frame_meta.frame_meta,
            self.ds_object_meta,
            parent.object_meta_impl.ds_object_meta if parent is not None else None,
        )

        self.ds_object_meta.confidence = confidence
        self._bbox = bbox  # cached BBox or RBBox structure
        nvds_set_obj_bbox(self._frame_meta.batch_meta, self.ds_object_meta, bbox)
        nvds_set_obj_uid(self._frame_meta.frame_meta, self.ds_object_meta)

    @property
    def confidence(self) -> float:
        """Returns object confidence.

        :return: Object confidence.
        """
        return self.ds_object_meta.confidence

    @property
    def bbox(self) -> Union[BBox, RBBox]:
        """Returns bounding box of object.

        :return: object bounding box.
        """
        if self._bbox is None:
            self._bbox = nvds_get_obj_bbox(self.ds_object_meta)
        return self._bbox

    def sync_bbox(self):
        if self._bbox is not None and self._bbox.is_modified():
            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug('Syncing new bbox into DS meta %s', self._bbox)
            nvds_upd_obj_bbox(self.ds_object_meta, self._bbox)

    @property
    def uid(self) -> int:
        """Returns uid of the object."""
        return int(nvds_get_obj_uid(self._frame_meta, self.ds_object_meta))

    def get_attr_meta_list(
        self, element_name: str, attr_name: str
    ) -> Optional[List[AttributeMeta]]:
        """Returns attributes (multi-label case).

        :param element_name: Attribute model name.
        :param attr_name: Attribute name.
        :return: List of AttributeMeta or None if the object has no such attributes.
        """
        return nvds_get_obj_attr_meta_list(
            frame_meta=self._frame_meta,
            obj_meta=self.ds_object_meta,
            element_name=element_name,
            attr_name=attr_name,
        )

    def get_attr_meta(
        self, element_name: str, attr_name: str
    ) -> Optional[AttributeMeta]:
        """Returns attribute.

        :param element_name: Attribute model name.
        :param attr_name: Attribute name.
        :return: AttributeMeta or None if the object has no such attribute.
        """
        return nvds_get_obj_attr_meta(
            frame_meta=self._frame_meta,
            obj_meta=self.ds_object_meta,
            element_name=element_name,
            attr_name=attr_name,
        )

    def replace_attr_meta_list(
        self, element_name: str, attr_name: str, value: List[AttributeMeta]
    ):
        """Replaces attributes with a new list.

        :param element_name: Attribute model name.
        :param attr_name: Attribute name.
        :param value: List of AttributeMeta.
        """
        nvds_replace_obj_attr_meta_list(
            frame_meta=self._frame_meta,
            obj_meta=self.ds_object_meta,
            element_name=element_name,
            attr_name=attr_name,
            value=value,
        )

    def remove_attr_meta_list(self, element_name: str, attr_name: str):
        """Removes attributes.

        :param element_name: Attribute model name.
        :param attr_name: Attribute name.
        """
        nvds_remove_obj_attr_meta_list(
            frame_meta=self._frame_meta,
            obj_meta=self.ds_object_meta,
            element_name=element_name,
            attr_name=attr_name,
        )

    def add_attr_meta(
        self,
        element_name: str,
        name: str,
        value: Any,
        confidence: float = 1.0,
        replace: bool = False,
    ):
        """Adds attribute to the object.

        :param element_name: Attribute model name.
        :param name: Attribute name.
        :param value: Attribute value.
        :param confidence: Attribute confidence.
        :param replace: Replace attribute if it already exists.
        """
        nvds_add_attr_meta_to_obj(
            frame_meta=self._frame_meta,
            obj_meta=self.ds_object_meta,
            element_name=element_name,
            name=name,
            value=value,
            confidence=confidence,
            replace=replace,
        )

    @property
    def label(self) -> str:
        """Returns the object label.

        :return: Object label.
        """
        _, object_label = parse_compound_key(self.ds_object_meta.obj_label)
        return object_label

    @label.setter
    def label(self, value: str):
        """Sets object label.

        :param value: Object label.
        """
        if isinstance(value, str):
            obj_key = build_model_object_key(self.element_name, value)
            if len(obj_key) > MAX_LABEL_SIZE:
                self.logger.warning(
                    'The length of label "%s" '
                    'and element_name "%s" is greater '
                    'than %d characters, '
                    'so it will be reduced '
                    'to %d characters',
                    value,
                    self.element_name,
                    MAX_LABEL_SIZE,
                    MAX_LABEL_SIZE,
                )

                self.ds_object_meta.obj_label = obj_key[:MAX_LABEL_SIZE]
            else:
                self.ds_object_meta.obj_label = obj_key
        else:
            raise MetaValueError(
                'The label property can only be a string, '
                f'the value is `{type(value)}`'
            )

    @property
    def draw_label(self) -> str:
        draw_label = nvds_get_obj_draw_label(self.ds_object_meta)
        if draw_label is not None:
            return draw_label
        return self.label

    @draw_label.setter
    def draw_label(self, value: str):
        if isinstance(value, str):
            ret = nvds_set_obj_draw_label(self.ds_object_meta, value)
            if not ret:
                nvds_init_obj_draw_label(
                    self._frame_meta.base_meta.batch_meta,
                    self.ds_object_meta,
                    value,
                )
        else:
            raise MetaValueError(
                'The label property can only be a string, '
                f'the value is `{type(value)}`'
            )

    @property
    def parent(self) -> '_NvDsObjectMetaImpl':
        """Returns this object's parent.

        :return: Parent object.
        """
        return self._parent_object

    @parent.setter
    def parent(self, value: Union['_NvDsObjectMetaImpl', ObjectMeta, None]):
        """Sets this object's parent.

        :param value: Parent object.
        """
        if value is None:
            self._parent_object = None
            self.ds_object_meta.parent = None
        elif isinstance(value, ObjectMeta) and isinstance(
            value.object_meta_impl, _NvDsObjectMetaImpl
        ):
            self._parent_object = value.object_meta_impl
            self.ds_object_meta.parent = value.object_meta_impl.ds_object_meta
        elif isinstance(value, _NvDsObjectMetaImpl):
            self._parent_object = value
            self.ds_object_meta.parent = value.ds_object_meta
        else:
            raise MetaValueError(
                f'{self.__class__.__name__} supports only '
                f'{self.__class__.__name__} as a parent object.'
            )

        if self._parent_object and self._parent_object.uid == self.uid:
            raise MetaValueError('An object cannot have itself as a parent.')

    @property
    def track_id(self) -> int:
        """Returns unique ID to track the object.
        :py:const:`~savant.meta.constants.UNTRACKED_OBJECT_ID` indicates
        the object has not been tracked.

        :return: unique ID for object.
        """
        return self.ds_object_meta.object_id

    @track_id.setter
    def track_id(self, value: int):
        """Sets unique ID to track the object.
        :py:const:`~savant.meta.constants.UNTRACKED_OBJECT_ID` indicates
        the object has not been tracked.

        :param value: Unique int ID.
        """
        self.ds_object_meta.object_id = value

    @property
    def element_name(self) -> str:
        """Returns the identifier of the element that created this object."""
        return get_model_name(model_id=self.ds_object_meta.unique_component_id)

    @classmethod
    def from_nv_ds_object_meta(
        cls,
        object_meta: pyds.NvDsObjectMeta,
        frame_meta: pyds.NvDsFrameMeta,
        depth: int = 2,
    ):
        """Factory method, creates instance of this class from pyds meta.

        :param object_meta: Deepstream object meta.
        :param frame_meta: Deepstream frame meta.
        :param depth: Object parent recursion depth.
        :return:
        """
        self = cls.__new__(cls)
        self.ds_object_meta = object_meta
        self._frame_meta = frame_meta
        self._bbox = None
        if not nvds_is_empty_object_meta(object_meta.parent) and depth > 0:
            self._parent_object = _NvDsObjectMetaImpl.from_nv_ds_object_meta(
                object_meta.parent, frame_meta, depth - 1
            )
        else:
            self._parent_object = None
        return self

    def __eq__(self, other):
        if isinstance(other, _NvDsObjectMetaImpl):
            if self.ds_object_meta == other.ds_object_meta:
                return True
        return False
