import functools
from subprocess import Popen, TimeoutExpired
from typing import Optional

from savant.gstreamer import Gst
from savant.gstreamer.codecs import Codec
from savant.utils.logging import get_logger

logger = get_logger('adapters.ao_sink.utils')


@functools.lru_cache
def nvidia_runtime_is_available() -> bool:
    """Check if Nvidia runtime is available."""

    return Gst.ElementFactory.find('nvv4l2h264enc') is not None


def process_is_alive(process: Popen) -> Optional[int]:
    try:
        return process.wait(0.01)
    except TimeoutExpired:
        pass


@functools.lru_cache
def check_codec_is_available(codec: Codec) -> bool:
    if nvidia_runtime_is_available():
        logger.info(
            'NVIDIA runtime is available. Using hardware-based decoding/encoding.'
        )
        if not codec.value.nv_encoder:
            logger.error(
                'Hardware-based encoding is not available for codec %s.',
                codec.value.name,
            )
            return False

        from savant.utils.check_display import check_display_env

        check_display_env(logger)

        from savant.deepstream.encoding import check_encoder_is_available

        return check_encoder_is_available({'output_frame': {'codec': codec.value.name}})

    else:
        logger.warning(
            'You are using the AO-RTSP adapter with a software-based decoder/encoder '
            '(without NVDEC/NVENC). This mode must be used only when hardware-based '
            'encoding is not available (Jetson Orin Nano, A100, H100). '
            'If the hardware-based encoding is available, run the adapter with Nvidia '
            'runtime enabled to activate hardware-based decoding/encoding.'
        )
        if not codec.value.sw_encoder:
            logger.error(
                'Software-based encoding is not available for codec %s.',
                codec.value.name,
            )
            return False

    return True
