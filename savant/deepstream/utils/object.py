"""DeepStream object utils."""
from typing import Optional, Tuple, Union
import pyds
from savant_rs.primitives.geometry import BBox, RBBox
from pysavantboost import add_rbbox_to_object_meta, NvRBboxCoords, get_rbbox
from savant.meta.errors import (
    IncorrectSelectionType,
    UIDError,
    MetaPoolError,
    MetaValueError,
)
from savant.meta.constants import UNTRACKED_OBJECT_ID, DEFAULT_CONFIDENCE
from savant.meta.type import InformationType, ObjectSelectionType
from savant.deepstream.meta.constants import MAX_LABEL_SIZE
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
    confidence: float = DEFAULT_CONFIDENCE,
    obj_label: str = '',
    object_id: int = UNTRACKED_OBJECT_ID,
    parent: Optional[pyds.NvDsObjectMeta] = None,
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

    if selection_type == ObjectSelectionType.ROTATED_BBOX:
        bbox = RBBox(*bbox)
    elif selection_type == ObjectSelectionType.REGULAR_BBOX:
        bbox = BBox(*bbox[:-1])
    else:
        raise IncorrectBBoxType(f"Incorrect selection type '{selection_type}'")
    nvds_set_obj_bbox(batch_meta, obj_meta, bbox)
    nvds_set_obj_uid(frame_meta, obj_meta)
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
            bbox.xc,
            bbox.yc,
            bbox.width,
            bbox.height,
        )
    )


def nvds_get_obj_bbox(nvds_obj_meta: pyds.NvDsObjectMeta) -> Union[BBox, RBBox]:
    """Get the bounding box for specified object meta
    from Deepstream meta structures.

    :param nvds_obj_meta: NvDsObjectMeta.
    :return: BBox or RBBox.
    """
    bbox_type = nvds_get_obj_selection_type(nvds_obj_meta)

    if bbox_type == ObjectSelectionType.REGULAR_BBOX:
        if (
            nvds_obj_meta.tracker_bbox_info.org_bbox_coords.width > 0
            and nvds_obj_meta.tracker_bbox_info.org_bbox_coords.height > 0
        ):
            nv_ds_bbox = nvds_obj_meta.tracker_bbox_info.org_bbox_coords
        elif (
            nvds_obj_meta.detector_bbox_info.org_bbox_coords.width > 0
            and nvds_obj_meta.detector_bbox_info.org_bbox_coords.height > 0
        ):
            nv_ds_bbox = nvds_obj_meta.detector_bbox_info.org_bbox_coords
        else:
            raise MetaValueError(
                'No valid bounding box found in Deepstream meta information.'
            )
        return BBox.ltwh(
            nv_ds_bbox.left, nv_ds_bbox.top, nv_ds_bbox.width, nv_ds_bbox.height
        )

    if bbox_type == ObjectSelectionType.ROTATED_BBOX:
        rbbox = get_rbbox(nvds_obj_meta)
        if rbbox is None:
            raise MetaValueError(
                'No rotated bounding box found in Deepstream meta information.'
            )
        return RBBox(
            rbbox.x_center, rbbox.y_center, rbbox.width, rbbox.height, rbbox.angle
        )

    raise IncorrectSelectionType('Unsupported object selection type.')


def nvds_set_obj_bbox(
    batch_meta: pyds.NvDsBatchMeta,
    obj_meta: pyds.NvDsObjectMeta,
    bbox: Union[BBox, RBBox],
) -> None:
    """Set bbox values and object selection type for specified object meta
    into Deepstream meta structures."""
    if isinstance(bbox, BBox):
        nvds_set_aligned_bbox_for_obj_meta(obj_meta, bbox)
        nvds_set_obj_selection_type(
            obj_meta,
            ObjectSelectionType.REGULAR_BBOX,
        )

    elif isinstance(bbox, RBBox):
        rbbox_coords = NvRBboxCoords()
        rbbox_coords.x_center = bbox.xc
        rbbox_coords.y_center = bbox.yc
        rbbox_coords.width = bbox.width
        rbbox_coords.height = bbox.height
        rbbox_coords.angle = bbox.angle
        add_rbbox_to_object_meta(batch_meta, obj_meta, rbbox_coords)
        nvds_set_obj_selection_type(
            obj_meta,
            ObjectSelectionType.ROTATED_BBOX,
        )


def nvds_set_aligned_bbox_for_obj_meta(obj_meta: pyds.NvDsObjectMeta, bbox: BBox) -> None:
    """Set aligned bbox values for specified object meta
    into Deepstream meta structures."""
    bbox_coords = obj_meta.detector_bbox_info.org_bbox_coords
    bbox_coords.left = bbox.left
    bbox_coords.top = bbox.top
    bbox_coords.width = bbox.width
    bbox_coords.height = bbox.height

    rect_params = obj_meta.rect_params
    rect_params.left = bbox.left
    rect_params.top = bbox.top
    rect_params.width = bbox.width
    rect_params.height = bbox.height


def nvds_upd_obj_bbox(obj_meta: pyds.NvDsObjectMeta, bbox: Union[BBox, RBBox]):
    """Update bbox values for specified object meta
    into Deepstream meta structures."""
    if isinstance(bbox, BBox):
        nvds_set_aligned_bbox_for_obj_meta(obj_meta, bbox)

    elif isinstance(bbox, RBBox):
        rbbox = get_rbbox(obj_meta)
        if rbbox is None:
            raise MetaValueError(
                'No rotated bounding box found in Deepstream meta information.'
            )
        rbbox.x_center = bbox.xc
        rbbox.y_center = bbox.yc
        rbbox.width = bbox.width
        rbbox.height = bbox.height
        rbbox.angle = bbox.angle


def nvds_init_obj_draw_label(
    batch_meta: pyds.NvDsBatchMeta, obj_meta: pyds.NvDsObjectMeta, draw_label: str
):
    """Initialize Deepstream meta structure
    for object draw label for specified object meta."""
    user_meta = pyds.nvds_acquire_user_meta_from_pool(batch_meta)
    if user_meta:
        data = pyds.alloc_custom_struct(user_meta)
        data.message = draw_label
        user_meta.user_meta_data = data
        user_meta.base_meta.meta_type = OBJ_DRAW_LABEL_META_TYPE
        pyds.nvds_add_user_meta_to_obj(obj_meta, user_meta)
    else:
        raise MetaPoolError('Error in acquiring user meta from pool.')


def nvds_get_obj_draw_label_struct(
    obj_meta: pyds.NvDsObjectMeta,
) -> pyds.CustomDataStruct:
    """Get Deepstream meta structure for object draw label for specified object meta."""
    for user_meta in nvds_obj_user_meta_iterator(obj_meta):
        if user_meta.base_meta.meta_type == OBJ_DRAW_LABEL_META_TYPE:
            return pyds.CustomDataStruct.cast(user_meta.user_meta_data)
    return None


def nvds_set_obj_draw_label(obj_meta: pyds.NvDsObjectMeta, draw_label: str) -> bool:
    """Set object draw label for specified object meta."""
    data = nvds_get_obj_draw_label_struct(obj_meta)
    if data is not None:
        data.message = draw_label
        return True
    return False


def nvds_get_obj_draw_label(obj_meta: pyds.NvDsObjectMeta) -> Optional[str]:
    """Get object draw label for specified object meta. Returns None if not set."""
    data = nvds_get_obj_draw_label_struct(obj_meta)
    if data is not None:
        return pyds.get_string(data.message)
    return None
