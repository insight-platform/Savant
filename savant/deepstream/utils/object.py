"""DeepStream object utils."""
from typing import Optional, Tuple, Union
import pyds
from pysavantboost import add_rbbox_to_object_meta, NvRBboxCoords
from savant.meta.errors import IncorrectSelectionType, UIDError, MetaPoolError
from savant.meta.constants import UNTRACKED_OBJECT_ID, DEFAULT_CONFIDENCE
from savant.meta.type import InformationType, ObjectSelectionType
from savant.deepstream.meta.constants import MAX_LABEL_SIZE
from savant.deepstream.meta.bbox import NvDsBBox, NvDsRBBox
from .iterator import nvds_obj_user_meta_iterator
from .meta_types import OBJ_DRAW_LABEL_META_TYPE


class IncorrectBBoxType(Exception):
    """Exception on errors when working with type of object on frame."""


def nvds_add_obj_meta_to_frame(  # pylint: disable=too-many-arguments,too-many-locals
    batch_meta: pyds.NvDsBatchMeta,
    frame_meta: pyds.NvDsFrameMeta,
    selection_type: int,
    class_id: int,
    gie_uid: int,
    bbox: Tuple[float, float, float, float, float],
    object_id: int = UNTRACKED_OBJECT_ID,
    parent: Optional[pyds.NvDsObjectMeta] = None,
    obj_label: str = '',
    confidence: float = DEFAULT_CONFIDENCE,
) -> pyds.NvDsObjectMeta:
    """Adds object meta to frame.

    :param batch_meta: NvDsBatchMeta to acquire object meta from.
    :param frame_meta: NvDsFrameMeta to add object meta to.
    :param selection_type: Object selection type.
    :param class_id: object class.
    :param gie_uid: inference engine id, `gie-unique-id`.
    :param bbox: tuple(x_center, y_center, width, height, angle)
        holds the bbox's rect_params in pixels, float, and angle in degrees, float.
    :param object_id: Id to assign to object.
        In DS max uint64 value used for untracked objects.
    :param parent: the parent NvDsObjectMeta object.
    :param obj_label: string describing the class of the detected object.
    :param confidence: detector confidence, float.
    """
    obj_meta = pyds.nvds_acquire_obj_meta_from_pool(batch_meta)

    obj_meta.class_id = class_id
    obj_meta.unique_component_id = gie_uid  # crucial for inference on specific gie
    obj_meta.object_id = object_id
    if obj_label:
        obj_meta.obj_label = obj_label[:MAX_LABEL_SIZE]  # MAX_LABEL_SIZE

    obj_meta.confidence = confidence

    x_center, y_center, width, height, angle = bbox

    if selection_type == ObjectSelectionType.ROTATED_BBOX:
        rbbox_coords = NvRBboxCoords()
        rbbox_coords.x_center = x_center
        rbbox_coords.y_center = y_center
        rbbox_coords.width = width
        rbbox_coords.height = height
        rbbox_coords.angle = angle
        add_rbbox_to_object_meta(batch_meta, obj_meta, rbbox_coords)
        nvds_set_obj_selection_type(obj_meta=obj_meta, selection_type=selection_type)
    elif selection_type == ObjectSelectionType.REGULAR_BBOX:
        bbox_coords = obj_meta.detector_bbox_info.org_bbox_coords
        bbox_coords.left = x_center - width / 2
        bbox_coords.top = y_center - height / 2
        bbox_coords.width = width
        bbox_coords.height = height

        rect_params = obj_meta.rect_params
        rect_params.left = x_center - width / 2
        rect_params.top = y_center - height / 2
        rect_params.width = width
        rect_params.height = height

        nvds_set_obj_selection_type(obj_meta=obj_meta, selection_type=selection_type)
    else:
        raise IncorrectBBoxType(f"Incorrect selection type '{selection_type}'")
    nvds_set_obj_uid(frame_meta=frame_meta, obj_meta=obj_meta)
    pyds.nvds_add_obj_meta_to_frame(frame_meta, obj_meta, parent)
    return obj_meta


def nvds_set_obj_selection_type(obj_meta: pyds.NvDsObjectMeta, selection_type: int):
    """Sets the type of object selection.

    :param obj_meta: deepstream object meta
    :param selection_type: selection type
    """

    obj_meta.misc_obj_info[InformationType.OBJECT_SELECTION_TYPE] = selection_type


def nvds_get_obj_selection_type(obj_meta: pyds.NvDsObjectMeta) -> int:
    """Gets the type of object selection.

    :param obj_meta: deepstream object meta
    :return: Selection type
    """

    return obj_meta.misc_obj_info[InformationType.OBJECT_SELECTION_TYPE]


def nvds_set_obj_uid(
    frame_meta: pyds.NvDsFrameMeta, obj_meta: pyds.NvDsObjectMeta
) -> int:
    """Sets the unique id for object.

    :param frame_meta: deepstream frame meta
    :param obj_meta: deepstream object meta
    :return: The set uid for the object
    """
    if obj_meta.misc_obj_info[InformationType.OBJECT_HASH_KEY]:
        raise UIDError('The object already has a unique key')
    obj_uid = nvds_generate_obj_uid(frame_meta, obj_meta)
    obj_meta.misc_obj_info[InformationType.OBJECT_HASH_KEY] = obj_uid

    return obj_uid


def nvds_get_obj_uid(
    frame_meta: pyds.NvDsFrameMeta, obj_meta: pyds.NvDsObjectMeta
) -> int:
    """Returns a unique object id. If the object does not have a unique
    identifier it will be set.

    :param frame_meta: NvDsFrameMeta.
    :param obj_meta: NvDsObjectMeta.
    :return: unique object id.
    """
    obj_uid = obj_meta.misc_obj_info[InformationType.OBJECT_HASH_KEY]
    if obj_uid == 0:
        if not nvds_get_obj_selection_type(obj_meta):
            nvds_set_obj_selection_type(obj_meta, ObjectSelectionType.REGULAR_BBOX)
        obj_uid = nvds_set_obj_uid(frame_meta=frame_meta, obj_meta=obj_meta)
    return obj_uid


def nvds_generate_obj_uid(
    frame_meta: pyds.NvDsFrameMeta, obj_meta: pyds.NvDsObjectMeta
) -> int:
    """Generates a unique id for object.

    :param frame_meta: NvDsFrameMeta.
    :param obj_meta: NvDsObjectMeta.
    :return: unique object id.
    """
    bbox = nvds_get_obj_bbox(obj_meta)
    return hash(
        (
            frame_meta.source_id,
            frame_meta.frame_num,
            obj_meta.obj_label,
            bbox.x_center,
            bbox.y_center,
            bbox.width,
            bbox.height,
        )
    )


def nvds_get_obj_bbox(nvds_obj_meta: pyds.NvDsFrameMeta) -> Union[NvDsBBox, NvDsRBBox]:
    """Returns BBox instance for specified frame meta.

    :param nvds_obj_meta: NvDsObjectMeta.
    :return:
    """
    if nvds_get_obj_selection_type(nvds_obj_meta) == ObjectSelectionType.REGULAR_BBOX:
        return NvDsBBox(nvds_obj_meta)

    if nvds_get_obj_selection_type(nvds_obj_meta) == ObjectSelectionType.ROTATED_BBOX:
        return NvDsRBBox(nvds_obj_meta)

    raise IncorrectSelectionType('Unsupported object selection type.')


def nvds_init_obj_draw_label(
    batch_meta: pyds.NvDsBatchMeta, obj_meta: pyds.NvDsObjectMeta, draw_label: str
):
    user_meta = pyds.nvds_acquire_user_meta_from_pool(batch_meta)
    if user_meta:
        data = pyds.alloc_custom_struct(user_meta)
        data.message = draw_label
        user_meta.user_meta_data = data
        user_meta.base_meta.meta_type = OBJ_DRAW_LABEL_META_TYPE
        pyds.nvds_add_user_meta_to_obj(obj_meta, user_meta)
    else:
        raise MetaPoolError('Error in acquiring user meta from pool.')


def nvds_get_obj_draw_label_struct(obj_meta: pyds.NvDsObjectMeta):
    for user_meta in nvds_obj_user_meta_iterator(obj_meta):
        if user_meta.base_meta.meta_type == OBJ_DRAW_LABEL_META_TYPE:
            return pyds.CustomDataStruct.cast(user_meta.user_meta_data)
    return None


def nvds_set_obj_draw_label(obj_meta: pyds.NvDsObjectMeta, draw_label: str) -> bool:
    data = nvds_get_obj_draw_label_struct(obj_meta)
    if data is not None:
        data.message = draw_label
        return True
    return False


def nvds_get_obj_draw_label(obj_meta: pyds.NvDsObjectMeta) -> Optional[str]:
    data = nvds_get_obj_draw_label_struct(obj_meta)
    if data is not None:
        return pyds.get_string(data.message)
    return None
