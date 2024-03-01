import functools
from subprocess import Popen, TimeoutExpired
from typing import Optional

from savant.gstreamer import Gst


@functools.lru_cache
def nvidia_runtime_is_available() -> bool:
    """Check if Nvidia runtime is available."""

    return Gst.ElementFactory.find('nvv4l2h264enc') is not None


def process_is_alive(process: Popen) -> Optional[int]:
    try:
        return process.wait(0.01)
    except TimeoutExpired:
        pass
