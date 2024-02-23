import functools

from savant.gstreamer import Gst


@functools.lru_cache
def nvidia_runtime_is_available() -> bool:
    """Check if Nvidia runtime is available."""

    return Gst.ElementFactory.find('nvv4l2h264enc') is not None
