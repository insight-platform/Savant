"""DeepStream object attribute utils."""
from typing import Any, Dict, Iterable, List, Optional, Tuple
import pyds
from savant.meta.attribute import AttributeMeta
from .object import nvds_get_obj_uid


# attribute storage, workaround
NVDS_OBJ_ATTR_STORAGE: Dict[int, Dict[Tuple[str, str], List[AttributeMeta]]] = {}


def nvds_add_attr_meta_to_obj(  # pylint: disable=too-many-arguments
    frame_meta: pyds.NvDsFrameMeta,
    obj_meta: pyds.NvDsObjectMeta,
    element_name: str,
    name: str,
    value: Any,
    confidence: float = 1.0,
):
    """Adds attribute to the object.

    :param frame_meta: object parent frame.
    :param obj_meta: object metadata.
    :param element_name: element name that created this attribute.
    :param name: attribute name.
    :param value: attribute value.
    :param confidence: object confidence.
    """
    skey = nvds_get_obj_uid(frame_meta, obj_meta)
    if skey not in NVDS_OBJ_ATTR_STORAGE:
        NVDS_OBJ_ATTR_STORAGE[skey] = {}
    if (element_name, name) not in NVDS_OBJ_ATTR_STORAGE[skey]:
        NVDS_OBJ_ATTR_STORAGE[skey][(element_name, name)] = []
    NVDS_OBJ_ATTR_STORAGE[skey][(element_name, name)].append(
        AttributeMeta(
            element_name=element_name, name=name, value=value, confidence=confidence
        )
    )


def nvds_attr_meta_iterator(
    frame_meta: pyds.NvDsFrameMeta,
    obj_meta: pyds.NvDsObjectMeta,
) -> Iterable[List[AttributeMeta]]:
    """AttributeMeta iterator(iterable).

    :param frame_meta: object parent frame.
    :param obj_meta: object metadata.
    :return: object attributes.
    """
    skey = nvds_get_obj_uid(frame_meta, obj_meta)
    if skey not in NVDS_OBJ_ATTR_STORAGE:
        return []
    return NVDS_OBJ_ATTR_STORAGE[skey].values()


def nvds_get_obj_attr_meta_list(
    frame_meta: pyds.NvDsFrameMeta,
    obj_meta: pyds.NvDsObjectMeta,
    model_name: str,
    attr_name: str,
) -> Optional[List[AttributeMeta]]:
    """Returns specified object attribute values (multi-label case).

    :param frame_meta: object parent frame.
    :param obj_meta: object metadata.
    :param model_name: element name that created this attribute.
    :param attr_name: attribute name.
    :return: List of AttributeMeta/None
    """
    skey = nvds_get_obj_uid(frame_meta, obj_meta)
    if skey not in NVDS_OBJ_ATTR_STORAGE:
        return None
    return NVDS_OBJ_ATTR_STORAGE[skey].get((model_name, attr_name), None)


def nvds_get_obj_attr_meta(
    frame_meta: pyds.NvDsFrameMeta,
    obj_meta: pyds.NvDsObjectMeta,
    model_name: str,
    attr_name: str,
) -> Optional[AttributeMeta]:
    """Returns the first value (the first and only except in the case of a
    multi-label) for specified object attribute.

    :param frame_meta: object parent frame.
    :param obj_meta: object metadata.
    :param model_name: element name that created this attribute.
    :param attr_name: attribute name.
    :return: AttributeMeta/None
    """
    attrs = nvds_get_obj_attr_meta_list(frame_meta, obj_meta, model_name, attr_name)
    return attrs[0] if attrs else None


def nvds_remove_obj_attrs(
    frame_meta: pyds.NvDsFrameMeta,
    obj_meta: pyds.NvDsObjectMeta,
):
    """Removes object attributes (from NVDS_OBJ_ATTR_STORAGE).

    :param frame_meta: object parent frame.
    :param obj_meta: object metadata.
    """
    skey = nvds_get_obj_uid(frame_meta, obj_meta)
    if skey in NVDS_OBJ_ATTR_STORAGE:
        del NVDS_OBJ_ATTR_STORAGE[skey]
