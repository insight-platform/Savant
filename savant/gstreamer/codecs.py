"""Gst codecs."""
from enum import Enum
from typing import Dict, List, NamedTuple, Optional


class CodecInfo(NamedTuple):
    """Codec info."""

    name: str
    """Codec name."""

    caps_name: str
    """Gstreamer caps."""

    caps_params: List[str]
    """Gstreamer caps params."""

    parser: Optional[str] = None
    """Gstreamer parser element."""

    encoder: Optional[str] = None
    """Gstreamer encoder element."""

    @property
    def caps_with_params(self) -> str:
        """Caps with caps params string."""
        return ','.join([self.caps_name] + self.caps_params)


class Codec(Enum):
    """Codec enum."""

    value: CodecInfo
    H264 = CodecInfo(
        'h264',
        'video/x-h264',
        ['stream-format=byte-stream', 'alignment=au'],
        'h264parse',
        'nvv4l2h264enc',
    )
    HEVC = CodecInfo(
        'hevc',
        'video/x-h265',
        ['stream-format=byte-stream', 'alignment=au'],
        'h265parse',
        'nvv4l2h265enc',
    )
    # TODO: add support for other raw formats (RGB, etc.)
    RAW_RGBA = CodecInfo('raw-rgba', 'video/x-raw', ['format=RGBA'])
    PNG = CodecInfo('png', 'image/png', [], 'pngparse', 'pngenc')
    JPEG = CodecInfo('jpeg', 'image/jpeg', [], 'jpegparse', 'jpegenc')


CODEC_BY_NAME: Dict[str, Codec] = {x.value.name: x for x in Codec}
CODEC_BY_CAPS_NAME: Dict[str, Codec] = {x.value.caps_name: x for x in Codec}
