import pyds

from savant.utils.platform import is_aarch64


def get_nvvideoconvert_properties():
    """Get nvvideoconvert properties based on platform."""

    if is_aarch64():
        return {'copy-hw': 2}  # VIC
    else:
        return {'nvbuf-memory-type': int(pyds.NVBUF_MEM_CUDA_UNIFIED)}
