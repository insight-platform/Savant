import ctypes

libnvdsgst_helper = ctypes.CDLL('libnvdsgst_helper.so')

# gst_nvevent_parse_pad_added
libnvdsgst_helper.gst_nvevent_parse_pad_added.argtypes = (
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_uint),
)
libnvdsgst_helper.gst_nvevent_parse_pad_added.restype = None

# gst_nvevent_parse_pad_deleted
libnvdsgst_helper.gst_nvevent_parse_pad_deleted.argtypes = (
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_uint),
)
libnvdsgst_helper.gst_nvevent_parse_pad_deleted.restype = None

# gst_nvevent_parse_stream_eos
libnvdsgst_helper.gst_nvevent_parse_stream_eos.argtypes = (
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_uint),
)
libnvdsgst_helper.gst_nvevent_parse_stream_eos.restype = None

# gst_nvevent_parse_stream_start
libnvdsgst_helper.gst_nvevent_parse_stream_start.argtypes = (
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_uint),
)
libnvdsgst_helper.gst_nvevent_parse_stream_start.restype = None
