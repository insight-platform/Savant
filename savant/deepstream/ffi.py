"""Ctypes foreign function interface for DeepStream structures and
functions."""
import ctypes
from savant.gstreamer.ffi import GST_PADDING


####################################
# NvBufSurface Types and Functions #
####################################


NVBUF_MAX_PLANES = 4


# pylint:disable=missing-class-docstring,too-few-public-methods
class NvBufSurfaceMappedAddr(ctypes.Structure):
    _fields_ = [
        ('addr', ctypes.c_void_p * NVBUF_MAX_PLANES),
        ('eglImage', ctypes.c_void_p),
        ('_reserved', ctypes.c_void_p * GST_PADDING),
    ]


class NvBufSurfacePlaneParams(ctypes.Structure):
    _fields_ = [
        ('num_planes', ctypes.c_uint32),
        ('width', ctypes.c_uint32 * NVBUF_MAX_PLANES),
        ('height', ctypes.c_uint32 * NVBUF_MAX_PLANES),
        ('pitch', ctypes.c_uint32 * NVBUF_MAX_PLANES),
        ('offset', ctypes.c_uint32 * NVBUF_MAX_PLANES),
        ('psize', ctypes.c_uint32 * NVBUF_MAX_PLANES),
        ('bytesPerPix', ctypes.c_uint32 * NVBUF_MAX_PLANES),
        ('_reserved', ctypes.c_void_p * (GST_PADDING * NVBUF_MAX_PLANES)),
    ]


class NvBufSurfaceParams(ctypes.Structure):
    _fields_ = [
        ('width', ctypes.c_uint32),
        ('height', ctypes.c_uint32),
        ('pitch', ctypes.c_uint32),
        ('colorFormat', ctypes.c_int),
        ('layout', ctypes.c_int),
        ('bufferDesc', ctypes.c_uint64),
        ('dataSize', ctypes.c_uint32),
        ('dataPtr', ctypes.c_void_p),
        ('planeParams', NvBufSurfacePlaneParams),
        ('mappedAddr', NvBufSurfaceMappedAddr),
        ('_reserved', ctypes.c_void_p * GST_PADDING),
    ]


class NvBufSurface(ctypes.Structure):
    _fields_ = [
        ('gpuId', ctypes.c_uint32),
        ('batchSize', ctypes.c_uint32),
        ('numFilled', ctypes.c_uint32),
        ('isContiguous', ctypes.c_bool),
        ('memType', ctypes.c_int),
        ('surfaceList', ctypes.POINTER(NvBufSurfaceParams)),
        ('_gst_reserved', ctypes.c_void_p * GST_PADDING),
    ]


LIBNVBUFSURFACE = ctypes.CDLL('libnvbufsurface.so')
LIBNVBUFSURFACE.NvBufSurfaceMap.argtypes = [
    ctypes.POINTER(NvBufSurface),
    # index - Index of a buffer in the batch (frame_meta.batch_id), -1 for all
    ctypes.c_int,
    # plane - Index of a plane in the buffer (0?), -1 for all
    ctypes.c_int,
    # NvBufSurfaceMemMapFlags type, 0 READ, 1 WRITE, 2 READ_WRITE
    ctypes.c_int,
]
LIBNVBUFSURFACE.NvBufSurfaceMap.restype = (
    ctypes.c_int
)  # 0 if successful, or -1 otherwise

LIBNVBUFSURFACE.NvBufSurfaceUnMap.argtypes = [
    ctypes.POINTER(NvBufSurface),
    # index - Index of a buffer in the batch (frame_meta.batch_id), -1 for all
    ctypes.c_int,
    # plane - Index of a plane in the buffer (0?), -1 for all
    ctypes.c_int,
]
LIBNVBUFSURFACE.NvBufSurfaceUnMap.restype = (
    ctypes.c_int
)  # 0 if successful, or -1 otherwise

LIBNVBUFSURFACE.NvBufSurfaceSyncForCpu.argtypes = [
    ctypes.POINTER(NvBufSurface),
    # index - Index of a buffer in the batch (frame_meta.batch_id), -1 for all
    ctypes.c_int,
    # plane - Index of a plane in the buffer (0?), -1 for all
    ctypes.c_int,
]
LIBNVBUFSURFACE.NvBufSurfaceSyncForCpu.restype = (
    ctypes.c_int
)  # 0 if successful, or -1 otherwise

LIBNVBUFSURFACE.NvBufSurfaceSyncForDevice.argtypes = [
    ctypes.POINTER(NvBufSurface),
    # index - Index of a buffer in the batch (frame_meta.batch_id), -1 for all
    ctypes.c_int,
    # plane - Index of a plane in the buffer (0?), -1 for all
    ctypes.c_int,
]
LIBNVBUFSURFACE.NvBufSurfaceSyncForDevice.restype = (
    ctypes.c_int
)  # 0 if successful, or -1 otherwise


###########################
# Latency Measurement API #
###########################


class NvDsFrameLatencyInfo(ctypes.Structure):
    _fields_ = [
        ('source_id', ctypes.c_uint),
        ('frame_num', ctypes.c_uint),
        ('comp_in_timestamp', ctypes.c_double),
        ('latency', ctypes.c_double),
    ]


LIBNVDSMETA = ctypes.CDLL('libnvdsgst_meta.so')

LIBNVDSMETA.nvds_get_enable_latency_measurement.argtypes = []
LIBNVDSMETA.nvds_get_enable_latency_measurement.restype = ctypes.c_int  # bool

LIBNVDSMETA.nvds_measure_buffer_latency.argtypes = [
    # A pointer to a Gst Buffer to which NvDsBatchMeta is attached as metadata.
    ctypes.c_void_p,
    # A pointer to an NvDsFrameLatencyInfo structure allocated for a batch of this size.
    # The function fills it with information about all of the sources.
    ctypes.POINTER(NvDsFrameLatencyInfo),
]
LIBNVDSMETA.nvds_measure_buffer_latency.restype = (
    ctypes.c_uint
)  # number of sources in batch
