"""Gst codecs."""
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

from savant.gstreamer import Gst

logger = logging.getLogger(__name__)


@dataclass
class CodecInfo:
    """Codec info."""

    name: str
    """Codec name."""

    caps_name: str
    """Gstreamer caps."""

    caps_params: List[str]
    """Gstreamer caps params."""

    parser: Optional[str] = None
    """Gstreamer parser element."""

    encoder_elements: Optional[List[str]] = None
    """Gstreamer encoder elements.
    Savant will use the first available element.
    """

    @property
    def caps_with_params(self) -> str:
        """Caps with caps params string."""
        return ','.join([self.caps_name] + self.caps_params)

    @property
    def encoder(self) -> Optional[str]:
        if not hasattr(self, '_encoder'):
            if not self.encoder_elements:
                encoder = None
            elif len(self.encoder_elements) == 1:
                encoder = self.encoder_elements[0]
            else:
                for element_name in self.encoder_elements:
                    logger.debug('Check if element %r exists', element_name)
                    elem_factory = Gst.ElementFactory.find(element_name)
                    if elem_factory is not None:
                        logger.debug('Found element %r', element_name)
                        encoder = element_name
                        break
                    logger.debug('Element %r not found', element_name)
                else:
                    encoder = self.encoder_elements[-1]

            self._encoder = encoder

        return self._encoder


class Codec(Enum):
    """Codec enum."""

    value: CodecInfo
    H264 = CodecInfo(
        'h264',
        'video/x-h264',
        ['stream-format=byte-stream', 'alignment=au'],
        'h264parse',
        ['nvv4l2h264enc'],
    )
    HEVC = CodecInfo(
        'hevc',
        'video/x-h265',
        ['stream-format=byte-stream', 'alignment=au'],
        'h265parse',
        ['nvv4l2h265enc'],
    )
    # TODO: add support for other raw formats (RGB, etc.)
    RAW_RGBA = CodecInfo('raw-rgba', 'video/x-raw', ['format=RGBA'])
    PNG = CodecInfo('png', 'image/png', [], 'pngparse', ['pngenc'])
    JPEG = CodecInfo('jpeg', 'image/jpeg', [], 'jpegparse', ['nvjpegenc', 'jpegenc'])


CODEC_BY_NAME: Dict[str, Codec] = {x.value.name: x for x in Codec}
CODEC_BY_CAPS_NAME: Dict[str, Codec] = {x.value.caps_name: x for x in Codec}
