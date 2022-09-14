"""Ctypes foreign function interface for GStreamer structures and functions."""
import ctypes

GST_PADDING = 4


# pylint:disable=missing-class-docstring,too-few-public-methods
class GstMapInfo(ctypes.Structure):
    """GstMapInfo structure map."""

    _fields_ = [
        ('memory', ctypes.c_void_p),
        ('flags', ctypes.c_int),
        ('data', ctypes.POINTER(ctypes.c_ubyte)),
        ('size', ctypes.c_size_t),
        ('maxsize', ctypes.c_size_t),
        ('user_data', ctypes.c_void_p * 4),
        ('_gst_reserved', ctypes.c_void_p * GST_PADDING),
    ]


LIBGST = ctypes.CDLL('libgstreamer-1.0.so.0')
LIBGST.gst_buffer_map.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(GstMapInfo),
    ctypes.c_int,
]
LIBGST.gst_buffer_map.restype = ctypes.c_bool

LIBGST.gst_buffer_unmap.argtypes = [ctypes.c_void_p, ctypes.POINTER(GstMapInfo)]
LIBGST.gst_buffer_unmap.restype = None

LIBGST.gst_mini_object_is_writable.argtypes = [ctypes.c_void_p]
LIBGST.gst_mini_object_is_writable.restype = ctypes.c_bool
