"""Object meta."""

from abc import ABC, abstractmethod
from typing import Any, Iterable, List, Optional, Union

from savant_rs.primitives.geometry import BBox, RBBox

from savant.meta.attribute import AttributeMeta
from savant.meta.constants import (
    DEFAULT_CONFIDENCE,
    DEFAULT_MODEL_NAME,
    PRIMARY_OBJECT_LABEL,
    UNTRACKED_OBJECT_ID,
)
from savant.meta.errors import MetaValueError


class BaseObjectMetaImpl(ABC):
    """Base class to work with object meta of specific backend framework."""

    element_name: str
    label: str
    bbox: Union[BBox, RBBox]
    confidence: Optional[float] = DEFAULT_CONFIDENCE
    track_id: int = UNTRACKED_OBJECT_ID
    parent: Optional[Union['BaseObjectMetaImpl', 'ObjectMeta']] = None

    @abstractmethod
    def get_attr_meta_list(
        self, element_name: str, attr_name: str
    ) -> Optional[List[AttributeMeta]]:
        """Returns attributes (multi-label case).

        :param element_name: Attribute model name.
        :param attr_name: Attribute name.
        :return: List of AttributeMeta or None if the object has no such attributes.
        """

    @abstractmethod
    def get_attr_meta(
        self, element_name: str, attr_name: str
    ) -> Optional[AttributeMeta]:
        """Returns attribute.

        :param element_name: Attribute model name.
        :param attr_name: Attribute name.
        :return: AttributeMeta or None if the object has no such attribute.
        """

    @abstractmethod
    def replace_attr_meta_list(
        self, element_name: str, attr_name: str, value: List[AttributeMeta]
    ):
        """Replaces attributes with a new list.

        :param element_name: Attribute model name.
        :param attr_name: Attribute name.
        :param value: List of AttributeMeta.
        """

    @abstractmethod
    def remove_attr_meta_list(self, element_name: str, attr_name: str):
        """Removes attributes.

        :param element_name: Attribute model name.
        :param attr_name: Attribute name.
        """

    @abstractmethod
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


class ObjectMeta:
    """The ObjectMeta describes the object that was detected or created on the
    frame.

    :param element_name: The unique identifier of the component by which
        the object was created.
    :param label: Class label of the object.
    :param bbox: Bounding box of the object.
    :param confidence: Confidence of the object from detector.
    :param track_id: Unique ID for tracking the object.
        Default value is :py:const:`~savant.meta.constants.UNTRACKED_OBJECT_ID`.
        It indicates the object has not been tracked.
    :param parent: The parent object metadata.
    :param attributes: The list of object attributes.
    """

    def __init__(
        self,
        element_name: str,
        label: str,
        bbox: Union[BBox, RBBox],
        confidence: Optional[float] = DEFAULT_CONFIDENCE,
        track_id: int = UNTRACKED_OBJECT_ID,
        parent: Optional['ObjectMeta'] = None,
        attributes: Optional[Iterable[AttributeMeta]] = None,
        draw_label: Optional[str] = None,
    ):
        self._element_name = element_name
        self._label = label
        self._draw_label = draw_label
        self._confidence = confidence
        self._track_id = track_id
        self._parent = parent
        self._bbox = bbox
        self._uid = None
        self.object_meta_impl: Optional[BaseObjectMetaImpl] = None
        self._attributes = {}
        if attributes:
            for attr in attributes:
                self.add_attr_meta(
                    element_name=attr.element_name,
                    name=attr.name,
                    value=attr.value,
                    confidence=attr.confidence,
                )

    def get_attr_meta_list(
        self, element_name: str, attr_name: str
    ) -> Optional[List[AttributeMeta]]:
        """Returns attributes (multi-label case).

        :param element_name: Attribute model name.
        :param attr_name: Attribute name.
        :return: List of AttributeMeta or None if the object has no such attributes.
        """
        if self.object_meta_impl:
            return self.object_meta_impl.get_attr_meta_list(
                element_name=element_name, attr_name=attr_name
            )
        return self._attributes.get((element_name, attr_name), None)

    def get_attr_meta(
        self, element_name: str, attr_name: str
    ) -> Optional[AttributeMeta]:
        """Returns attribute.

        :param element_name: Attribute model name.
        :param attr_name: Attribute name.
        :return: AttributeMeta or None if the object has no such attribute.
        """
        attrs = self.get_attr_meta_list(element_name, attr_name)
        return attrs[0] if attrs else None

    def replace_attr_meta_list(
        self, element_name: str, attr_name: str, value: List[AttributeMeta]
    ):
        """Replaces attributes with a new list.

        :param element_name: Attribute model name.
        :param attr_name: Attribute name.
        :param value: List of AttributeMeta.
        """
        if self.object_meta_impl:
            self.object_meta_impl.replace_attr_meta_list(
                element_name=element_name, attr_name=attr_name, value=value
            )
        else:
            self._attributes[(element_name, attr_name)] = value

    def remove_attr_meta_list(self, element_name: str, attr_name: str):
        """Removes attributes.

        :param element_name: Attribute model name.
        :param attr_name: Attribute name.
        """
        if self.object_meta_impl:
            self.object_meta_impl.remove_attr_meta_list(
                element_name=element_name, attr_name=attr_name
            )
        elif (element_name, attr_name) in self._attributes:
            del self._attributes[(element_name, attr_name)]

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
        if self.object_meta_impl:
            self.object_meta_impl.add_attr_meta(
                element_name=element_name,
                name=name,
                value=value,
                confidence=confidence,
                replace=replace,
            )
        else:
            if not (element_name, name) in self._attributes:
                self._attributes[(element_name, name)] = []
            self._attributes[(element_name, name)].append(
                AttributeMeta(
                    element_name=element_name,
                    name=name,
                    value=value,
                    confidence=confidence,
                )
            )

    @property
    def label(self) -> str:
        """Returns the object label.

        :return: Object label.
        """
        if self.object_meta_impl:
            return self.object_meta_impl.label
        return self._label

    @label.setter
    def label(self, value: str):
        """Sets object label.

        :param value: Object label.
        """
        if self.object_meta_impl:
            self.object_meta_impl.label = value
        else:
            self._label = value

    @property
    def draw_label(self) -> str:
        if self.object_meta_impl:
            return self.object_meta_impl.draw_label
        if self._draw_label is not None:
            return self._draw_label
        return self.label

    @draw_label.setter
    def draw_label(self, value: str):
        if self.object_meta_impl:
            self.object_meta_impl.draw_label = value
        else:
            self._draw_label = value

    @property
    def parent(self) -> 'ObjectMeta':
        """Returns this object's parent.

        :return: Parent object.
        """
        if self.object_meta_impl and isinstance(
            self.object_meta_impl.parent, BaseObjectMetaImpl
        ):
            return ObjectMeta._from_be_object_meta(self.object_meta_impl.parent)
        return self._parent

    @parent.setter
    def parent(self, value: 'ObjectMeta'):
        """Sets this object's parent.

        :param value: Parent object.
        """
        if value.uid == self.uid:
            raise MetaValueError('An object cannot have itself as a parent.')
        if self.object_meta_impl:
            self.object_meta_impl.parent = value
        self._parent = value

    @property
    def track_id(self) -> int:
        """Returns unique ID to track the object.
        :py:const:`~savant.meta.constants.UNTRACKED_OBJECT_ID`
        indicates the object has not been tracked.

        :return: Unique ID for object.
        """
        if self.object_meta_impl:
            return self.object_meta_impl.track_id
        return self._track_id

    @track_id.setter
    def track_id(self, value: int):
        """Sets unique ID to track the object.
        :py:const:`~savant.meta.constants.UNTRACKED_OBJECT_ID`
        indicates the object has not been tracked.

        :param value: Unique int ID.
        """
        if self.object_meta_impl:
            self.object_meta_impl.track_id = value
        else:
            self._track_id = value

    @property
    def element_name(self) -> str:
        """Returns the identifier of the element that created this object."""
        if self.object_meta_impl:
            return self.object_meta_impl.element_name
        return self._element_name

    @element_name.setter
    def element_name(self, value: str):
        """Sets the identifier of the element that created this object.

        :param value: Unique integer element identifier.
        """
        if self.object_meta_impl:
            self.object_meta_impl = value
        self._element_name = value

    @property
    def is_primary(self) -> bool:
        return (
            self.element_name == DEFAULT_MODEL_NAME
            and self.label == PRIMARY_OBJECT_LABEL
        )

    @property
    def confidence(self) -> float:
        """Returns object confidence.

        :return: Object confidence.
        """
        if self.object_meta_impl:
            return self.object_meta_impl.confidence
        return self._confidence

    @property
    def bbox(self) -> Optional[Union[BBox, RBBox]]:
        """Returns bounding box of object or None if the object has no bounding
        box.

        :return: Instance of a class that implements the BoundingBox interface.
            It can be :py:class:`~savant.meta.bbox.RegularBoundingBox` or
            :py:class:`~savant.meta.bbox.RotatedBoundingBox`
        """
        if self.object_meta_impl:
            return self.object_meta_impl.bbox
        return self._bbox

    def sync_bbox(self):
        if self.object_meta_impl:
            self.object_meta_impl.sync_bbox()

    @property
    def uid(self) -> Optional[int]:
        """Returns uid of the object."""
        if self.object_meta_impl:
            return self.object_meta_impl.uid
        return self._uid

    @classmethod
    def _from_be_object_meta(cls, be_object_meta: BaseObjectMetaImpl):
        """Factory method, creates an instance of this class from some backend
        object meta implementation.

        :param be_object_meta: Backend object meta implementation.
        """
        self = cls.__new__(cls)
        self.object_meta_impl = be_object_meta
        return self

    def __eq__(self, other: 'ObjectMeta'):
        if isinstance(other, ObjectMeta):
            if self.object_meta_impl and other.object_meta_impl:
                return self.object_meta_impl == other.object_meta_impl
            if not self.object_meta_impl and not other.object_meta_impl:
                return self == other
        return False
