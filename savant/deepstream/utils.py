"""DeepStream utils."""
from contextlib import contextmanager
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    Iterable,
    List,
    Optional,
    Tuple,
    Union,
)
import ctypes
import pyds
import numpy as np
from pysavantboost import (
    add_rbbox_to_object_meta,
    NvRBboxCoords,
    get_rbbox,
)

from savant.deepstream.errors import NvDsBBoxError, NvDsRBBoxError
from savant.meta.errors import IncorrectSelectionType, UIDError
from savant.meta.attribute import AttributeMeta
from savant.meta.constants import UNTRACKED_OBJECT_ID, DEFAULT_CONFIDENCE
from savant.meta.type import InformationType, ObjectSelectionType
from savant.deepstream.meta.constants import MAX_LABEL_SIZE
from savant.gstreamer import Gst, GObject  # noqa:F401
from savant.gstreamer.ffi import GstMapInfo
from savant.gstreamer.utils import map_gst_buffer
from savant.deepstream.ffi import (
    LIBNVBUFSURFACE,
    NvBufSurface,
    LIBNVDSMETA,
    NvDsFrameLatencyInfo,
)


def _make_nvevent_type(event_type: int):
    return (event_type << Gst.EVENT_NUM_SHIFT) | (
        Gst.EventTypeFlags.DOWNSTREAM | Gst.EventTypeFlags.SERIALIZED
    )


# TODO: find or compile bindings for gst-nvevent.h
GST_NVEVENT_PAD_ADDED = _make_nvevent_type(400)
GST_NVEVENT_PAD_DELETED = _make_nvevent_type(401)
GST_NVEVENT_STREAM_EOS = _make_nvevent_type(402)
GST_NVEVENT_STREAM_SEGMENT = _make_nvevent_type(403)
GST_NVEVENT_STREAM_RESET = _make_nvevent_type(404)
GST_NVEVENT_STREAM_START = _make_nvevent_type(405)


class IncorrectBBoxType(Exception):
    """Exception on errors when working with type of object on frame."""


class NvDsMetaIterator:
    """NvDs meta data iterator.

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


def nvds_frame_iterator(
    buffer: Gst.Buffer, batch_meta: pyds.NvDsBatchMeta
) -> Generator[Tuple[pyds.NvDsFrameMeta, np.ndarray], None, None]:
    """NvDsFrameMeta iterator, returns also a frame as ndarray.

    :param buffer: gstreamer buffer that contains the batch_meta.
    :param batch_meta: NvDs batch metadata structure that will be iterated on.
    :return: NvDs frame metadata and frame image data.
    """
    with map_gst_buffer(buffer) as gst_map_info:
        for frame_meta in nvds_frame_meta_iterator(batch_meta):
            with map_nvbufsurface(gst_map_info, frame_meta.batch_id) as frame:
                yield frame_meta, frame


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
        nvds_set_selection_type(obj_meta=obj_meta, selection_type=selection_type)
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

        nvds_set_selection_type(obj_meta=obj_meta, selection_type=selection_type)
    else:
        raise IncorrectBBoxType(f"Incorrect selection type '{selection_type}'")
    nvds_set_obj_uid(frame_meta=frame_meta, obj_meta=obj_meta)
    pyds.nvds_add_obj_meta_to_frame(frame_meta, obj_meta, parent)
    return obj_meta


# attribute storage, workaround
NVDS_OBJ_ATTR_STORAGE: Dict[int, Dict[Tuple[str, str], List[AttributeMeta]]] = {}


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
        if not nvds_get_selection_type(obj_meta):
            nvds_set_selection_type(obj_meta, ObjectSelectionType.REGULAR_BBOX)
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
    if nvds_get_selection_type(obj_meta) == ObjectSelectionType.ROTATED_BBOX:
        rotated_bbox = nvds_get_rbbox(obj_meta)
        return hash(
            (
                frame_meta.source_id,
                frame_meta.frame_num,
                obj_meta.obj_label,
                rotated_bbox.x_center,
                rotated_bbox.y_center,
                rotated_bbox.width,
                rotated_bbox.height,
            )
        )
    if nvds_get_selection_type(obj_meta) == ObjectSelectionType.REGULAR_BBOX:
        bbox = obj_meta.rect_params
        if bbox.width == 0.0:
            bbox = obj_meta.detector_bbox_info.org_bbox_coords
        if bbox.width == 0.0:
            bbox = obj_meta.tracker_bbox_info.org_bbox_coords
        if bbox.width == 0.0:
            raise NvDsBBoxError(
                "Deepstream object meta doesn't contain valid bounding box "
                "and object uid can't be generated."
            )
        return hash(
            (
                frame_meta.source_id,
                frame_meta.frame_num,
                obj_meta.obj_label,
                bbox.left,
                bbox.top,
                bbox.width,
                bbox.height,
            )
        )

    raise IncorrectSelectionType(
        'Unsupported object selection type for hash key generation'
    )


def nvds_get_rbbox(nvds_obj_meta: pyds.NvDsFrameMeta) -> NvRBboxCoords:
    """Returns rotated bbox from user meta."""
    rbbox = get_rbbox(nvds_obj_meta)
    if rbbox:
        return rbbox

    raise NvDsRBBoxError(
        f'No rotated box is found for the object {nvds_obj_meta.object_id}'
    )


def nvds_add_attr_meta_to_obj(
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


def nvds_infer_tensor_meta_to_outputs(
    tensor_meta: pyds.NvDsInferTensorMeta, layer_names: List[str]
) -> List[Optional[np.ndarray]]:
    """Fetches output of specified layers from pyds.NvDsInferTensorMeta.

    :param tensor_meta: NvDsInferTensorMeta structure.
    :param layer_names: Names of layers to return, order is important.
    :return: List of specified layer data.
    """
    data_type_map = {
        pyds.NvDsInferDataType.FLOAT: ctypes.c_float,
        pyds.NvDsInferDataType.HALF: ctypes.c_float,  # TODO: Check
        pyds.NvDsInferDataType.INT8: ctypes.c_int8,
        pyds.NvDsInferDataType.INT32: ctypes.c_int32,
    }
    layers = [None] * len(layer_names)
    for i in range(tensor_meta.num_output_layers):
        layer_info = pyds.get_nvds_LayerInfo(tensor_meta, i)
        if not layer_info.isInput and layer_info.layerName in layer_names:
            layers[layer_names.index(layer_info.layerName)] = np.copy(
                np.ctypeslib.as_array(
                    ctypes.cast(
                        pyds.get_ptr(layer_info.buffer),
                        ctypes.POINTER(data_type_map[layer_info.dataType]),
                    ),
                    shape=layer_info.dims.d[: layer_info.dims.numDims],
                )
            )
    return layers


@contextmanager
def map_nvbufsurface(
    gst_map_info: GstMapInfo,
    batch_id: int,
    flag: int = pyds.NvBufSurfaceMemMapFlags.NVBUF_MAP_READ_WRITE,
) -> np.ndarray:
    """Gets a pointer to NvBufSurface from GstMapInfo and tries to map it.
    Unmaps at context exit.

    TODO: Check out pyds.get_nvds_buf_surface
    in order to replace custom implementation.
    There was a problem(?) using pyds.get_nvds_buf_surface with DS 5.1 on Jetson.

    :param gst_map_info: GstMapInfo structure of a mapped Gst.Buffer
    :param batch_id: batch id of a frame in Deepstream batch
    :param flag: an access mode flag from pyds.NvBufSurfaceMemMapFlags
    """
    nvbufsurface_p = ctypes.cast(gst_map_info.data, ctypes.POINTER(NvBufSurface))
    nvbufsurface = nvbufsurface_p.contents

    nvbufsurface_params = nvbufsurface.surfaceList[batch_id]
    nvbufsurface_mappedaddr = nvbufsurface_params.mappedAddr

    # TODO: Remove to avoid ValueError with second mapping?
    if nvbufsurface_mappedaddr.addr[0] is not None:
        raise ValueError('NvBufMappedAddr already filled.')

    ret_value = LIBNVBUFSURFACE.NvBufSurfaceMap(nvbufsurface_p, batch_id, 0, flag)
    if ret_value != 0:
        raise ValueError('Buffer map to be accessed by CPU failed.')

    # Deepstream C example `gstdsexample` calls this function,
    # but it always returns -1 (fail, according to docs)
    # https://forums.developer.nvidia.com/t/are-there-any-examples-for-pyds-nvbufsurface-objects-and-methods/124803
    # res = LIBNVBUFSURFACE.NvBufSurfaceSyncForCpu(nvbufsurface_p, batch_id, 0)

    try:
        shape = (
            nvbufsurface_params.planeParams.height[0],
            nvbufsurface_params.planeParams.width[0],
            nvbufsurface_params.planeParams.pitch[0]
            // nvbufsurface_params.planeParams.width[0],
        )
        ctypes_arr = ctypes.cast(
            nvbufsurface_mappedaddr.addr[0],
            ctypes.POINTER(ctypes.c_uint8 * shape[2] * shape[1] * shape[0]),
        ).contents
        yield np.ctypeslib.as_array(ctypes_arr)
    finally:
        # res = LIBNVBUFSURFACE.NvBufSurfaceSyncForDevice(
        #     nvbufsurface_p, frame_meta.batch_id, 0
        # )
        LIBNVBUFSURFACE.NvBufSurfaceUnMap(nvbufsurface_p, batch_id, 0)


def nvds_is_enabled_latency_measurement():
    """Indicates whether the environment variable
    NVDS_ENABLE_LATENCY_MEASUREMENT is exported."""
    return bool(LIBNVDSMETA.nvds_get_enable_latency_measurement())


def nvds_measure_buffer_latency(
    gst_buffer: Gst.Buffer, batch_size: int
) -> List[NvDsFrameLatencyInfo]:
    """Measures the latency of all frames present in the current batch. The
    latency is computed from decoder input up to the point this API is called.
    You can install the probe on either pad of the component and call this
    function to measure the latency.

    There are problems with using of NvDs Latency Measurement API

    1. unknown structure of nvds_batch_meta.batch_user_meta_list item with
       NVDS_LATENCY_MEASUREMENT_META type
    2. requirement to reverse batch_user_meta_list, see
       https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_Application_migration.html

    :param gst_buffer: gstreamer buffer that contains batch metadata.
    :param batch_size: number of frames in the batch.
    :return: NvDsFrameLatencyInfo structure.
    """
    gst_buffer_p = hash(gst_buffer)
    # allocate with a margin to avoid `Core dumped`
    latency_info: List[NvDsFrameLatencyInfo] = (
        NvDsFrameLatencyInfo * (batch_size + 1)
    )()
    num = LIBNVDSMETA.nvds_measure_buffer_latency(gst_buffer_p, latency_info)
    return latency_info[:num]


def _gst_nvevent_extract_source_id(event: Gst.Event, type, struct_name: str):
    if not (event.type == type and event.has_name(struct_name)):
        return None
    struct: Gst.Structure = event.get_structure()
    parsed, source_id = struct.get_uint('source-id')
    return source_id if parsed else None


# TODO: find or compile bindings for gst-nvevent.h
# pylint:disable=line-too-long
def gst_nvevent_parse_pad_deleted(event: Gst.Event) -> Optional[int]:
    """Extract source-id from GST_NVEVENT_PAD_DELETED.

    GST_NVEVENT_PAD_DELETED generated by nvstreammux.
    See https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvstreammux.html
    for details.
    """

    return _gst_nvevent_extract_source_id(
        event, GST_NVEVENT_PAD_DELETED, 'nv-pad-deleted'
    )


# TODO: find or compile bindings for gst-nvevent.h
# pylint:disable=line-too-long
def gst_nvevent_parse_stream_eos(event: Gst.Event) -> Optional[int]:
    """Extract source-id from GST_NVEVENT_STREAM_EOS.

    GST_NVEVENT_STREAM_EOS generated by nvstreammux.
    See https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvstreammux.html
    for details.
    """

    return _gst_nvevent_extract_source_id(
        event, GST_NVEVENT_STREAM_EOS, 'nv-stream-eos'
    )


# TODO: find or compile bindings for gst-nvevent.h
def gst_nvevent_new_stream_eos(source_id: int) -> Gst.Event:
    """Creates a "custom EOS" event for the specified source.

    :param source_id: Source ID of the stream for which EOS is to be sent;
                      also the pad ID  of the sinkpad of the
                      Gst-nvstreammux plugin for which
                      the source is configured.
    """

    struct: Gst.Structure = Gst.Structure.new_empty('nv-stream-eos')
    struct.set_value('source-id', GObject.Value(GObject.TYPE_UINT, source_id))
    event: Gst.Event = Gst.Event.new_custom(Gst.EventType.UNKNOWN, struct)
    event.type = GST_NVEVENT_STREAM_EOS
    return event


def nvds_set_selection_type(obj_meta: pyds.NvDsObjectMeta, selection_type: int):
    """Sets the type of object selection.

    :param obj_meta: deepstream object meta
    :param selection_type: selection type
    """

    obj_meta.misc_obj_info[InformationType.OBJECT_SELECTION_TYPE] = selection_type


def nvds_get_selection_type(
    obj_meta: pyds.NvDsObjectMeta,
) -> int:
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
