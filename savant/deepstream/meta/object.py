"""Deepstream-specific ObjectMeta interface implementation."""
from typing import Any, List, Optional, Union
import pyds
from pysavantboost import NvRBboxCoords, add_rbbox_to_object_meta

from savant.meta.bbox import BBox, RBBox
from savant.meta.errors import MetaValueError
from savant.deepstream.meta.bbox import NvDsBBox, NvDsRBBox
from savant.deepstream.meta.constants import MAX_LABEL_SIZE
from savant.deepstream.utils import (
    nvds_get_selection_type,
    nvds_get_obj_attr_meta,
    nvds_get_obj_attr_meta_list,
    nvds_add_attr_meta_to_obj,
    nvds_set_selection_type,
    nvds_set_obj_uid,
)
from savant.gstreamer.utils import LoggerMixin
from savant.meta.attribute import AttributeMeta
from savant.meta.constants import (
    UNTRACKED_OBJECT_ID,
    DEFAULT_CONFIDENCE,
)
from savant.meta.object import ObjectMeta, BaseObjectMetaImpl
from savant.meta.type import ObjectSelectionType
from savant.utils.model_registry import ModelObjectRegistry


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
    ):
        super().__init__()
        self._model_object_registry = ModelObjectRegistry()
        key = ModelObjectRegistry.model_object_key(element_name, label)
        element_uid, class_id = self._model_object_registry.get_model_object_ids(key)
        self._frame_meta = frame_meta

        self.ds_object_meta = pyds.nvds_acquire_obj_meta_from_pool(
            frame_meta.batch_meta
        )
        self.ds_object_meta.class_id = class_id
        self.ds_object_meta.unique_component_id = element_uid
        self.label = label  # MAX_LABEL_SIZE
        self.track_id = track_id
        self.parent = parent

        if (
            isinstance(parent, ObjectMeta)
            and isinstance(parent.object_meta_impl, _NvDsObjectMetaImpl)
            or parent is None
        ):
            pyds.nvds_add_obj_meta_to_frame(
                self._frame_meta.frame_meta,
                self.ds_object_meta,
                parent.object_meta_impl.ds_object_meta if parent is not None else None,
            )
        else:
            raise MetaValueError(
                f'parent is `{type(parent)}` type, expected `{self.__class__.__name__}`'
            )

        self.ds_object_meta.confidence = confidence
        if isinstance(bbox, BBox):
            bbox_coords = self.ds_object_meta.detector_bbox_info.org_bbox_coords
            bbox_coords.left = bbox.left
            bbox_coords.top = bbox.top
            bbox_coords.width = bbox.width
            bbox_coords.height = bbox.height

            rect_params = self.ds_object_meta.rect_params
            rect_params.left = bbox.left
            rect_params.top = bbox.top
            rect_params.width = bbox.width
            rect_params.height = bbox.height
            nvds_set_selection_type(
                obj_meta=self.ds_object_meta,
                selection_type=ObjectSelectionType.REGULAR_BBOX,
            )

        elif isinstance(bbox, RBBox):
            rbbox_coords = NvRBboxCoords()
            rbbox_coords.x_center = bbox.x_center
            rbbox_coords.y_center = bbox.y_center
            rbbox_coords.width = bbox.width
            rbbox_coords.height = bbox.height
            rbbox_coords.angle = bbox.angle
            add_rbbox_to_object_meta(
                self._frame_meta.batch_meta, self._frame_meta.frame_meta, rbbox_coords
            )
            nvds_set_selection_type(
                obj_meta=self.ds_object_meta,
                selection_type=ObjectSelectionType.ROTATED_BBOX,
            )
        nvds_set_obj_uid(
            frame_meta=self._frame_meta.frame_meta, obj_meta=self.ds_object_meta
        )

    @property
    def confidence(self) -> float:
        """Returns object confidence.

        :return: Object confidence.
        """
        return self.ds_object_meta.confidence

    @property
    def bbox(self) -> Optional[Union[BBox, RBBox]]:
        """Returns bounding box of object. Returns None if the object has no
        bounding box.

        :return: Instance one of class that implements the BoundingBox interface.
            It can be :py:class:`~savant.meta.bbox.RegularBoundingBox` or
            :py:class:`~savant.meta.bbox.RotatedBoundingBox`.
        """
        bbox_type = nvds_get_selection_type(self.ds_object_meta)
        if bbox_type == ObjectSelectionType.REGULAR_BBOX:
            return NvDsBBox(self.ds_object_meta)
        if bbox_type == ObjectSelectionType.ROTATED_BBOX:
            return NvDsRBBox(self.ds_object_meta)

    def get_attr_meta(
        self, element_name: str, attr_name: str
    ) -> Optional[AttributeMeta]:
        """Returns the specified attribute of the object.

        :param element_name: Attribute model name.
        :param attr_name: Attribute name.
        :return: AttributeMeta or None if the object has no such attribute.
        """
        return nvds_get_obj_attr_meta(
            frame_meta=self._frame_meta,
            obj_meta=self.ds_object_meta,
            model_name=element_name,
            attr_name=attr_name,
        )

    def get_attr_meta_list(
        self, element_name: str, attr_name: str
    ) -> Optional[List[AttributeMeta]]:
        """Returns a list of the object's specified attributes.

        :param element_name: Attribute model name.
        :param attr_name: Attribute name.
        :return: List of AttributeMeta or None if the object has no such attributes.
        """
        return nvds_get_obj_attr_meta_list(
            frame_meta=self._frame_meta,
            obj_meta=self.ds_object_meta,
            model_name=element_name,
            attr_name=attr_name,
        )

    def add_attr_meta(
        self,
        element_name: str,
        name: str,
        value: Any,
        confidence: float = 1.0,
    ):
        """Adds specified object attribute to object meta.

        :param element_name: attribute model name.
        :param name: attribute name.
        :param value: attribute value.
        :param confidence: attribute confidence.
        """
        nvds_add_attr_meta_to_obj(
            frame_meta=self._frame_meta,
            obj_meta=self.ds_object_meta,
            element_name=element_name,
            name=name,
            value=value,
            confidence=confidence,
        )

    @property
    def label(self) -> str:
        """Returns the object label.

        :return: Object label.
        """
        _, object_label = ModelObjectRegistry.parse_model_object_key(
            self.ds_object_meta.obj_label
        )
        return object_label

    @label.setter
    def label(self, value: str):
        """Sets object label.

        :param value: Object label.
        """
        if isinstance(value, str):
            obj_key = self._model_object_registry.model_object_key(
                self.element_name, value
            )
            if len(obj_key) > MAX_LABEL_SIZE:
                self.logger.warn(
                    f"The length of label '{value}' "
                    f'and element_name `{self.element_name}` is greater '
                    f'than {MAX_LABEL_SIZE} characters, '
                    f'so it will be reduced '
                    f'to {MAX_LABEL_SIZE} characters'
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
    def parent(self) -> '_NvDsObjectMetaImpl':
        """Returns this object's parent.

        :return: Parent object.
        """
        return self._parent_object

    @parent.setter
    def parent(self, value: Union['_NvDsObjectMetaImpl', ObjectMeta]):
        """Sets this object's parent.

        :param value: Parent object.
        """
        if value is None:
            self._parent_object = None
        elif isinstance(value, ObjectMeta) and isinstance(
            value.object_meta_impl, _NvDsObjectMetaImpl
        ):
            self._parent_object = value.object_meta_impl
        elif isinstance(value, _NvDsObjectMetaImpl):
            self._parent_object = value
        else:
            raise MetaValueError(
                f'{self.__class__.__name__} supports only '
                f'{self.__class__.__name__} as a parent object'
            )

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
        return self._model_object_registry.get_name(
            uid=self.ds_object_meta.unique_component_id
        )

    @classmethod
    def from_nv_ds_object_meta(
        cls, object_meta: pyds.NvDsObjectMeta, frame_meta: pyds.NvDsFrameMeta
    ):
        """Factory method, creates instance of this class from pyds meta.

        :param object_meta: Deepstream object meta.
        :param frame_meta: Deepstream frame meta.
        :return:
        """
        self = cls.__new__(cls)
        self._model_object_registry = ModelObjectRegistry()
        self.ds_object_meta = object_meta
        self._frame_meta = frame_meta
        if object_meta.parent:
            self._parent_object = _NvDsObjectMetaImpl.from_nv_ds_object_meta(
                object_meta=object_meta.parent, frame_meta=frame_meta
            )
        else:
            self._parent_object = None
        return self

    def __eq__(self, other):
        if isinstance(other, _NvDsObjectMetaImpl):
            if self.ds_object_meta == other.ds_object_meta:
                return True
        return False
