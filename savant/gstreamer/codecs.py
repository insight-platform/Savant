"""Gst codecs."""
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

from savant.gstreamer import Gst
from savant.utils.logging import get_logger

logger = get_logger(__name__)


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

    nv_encoder: Optional[str] = None
    """Nvenc gstreamer encoder element.
    Savant will use the it when encoder type is not specified.
    """

    sw_encoder: Optional[str] = None
    """Software gstreamer encoder element.
    Savant will use the it when encoder type is not specified and codec
    does not have NvEnc encoder.
    """

    is_raw: bool = False
    """Indicates if codec is raw.
    """

    @property
    def caps_with_params(self) -> str:
        """Caps with caps params string."""
        return ','.join([self.caps_name] + self.caps_params)

    def encoder(self, encoder_type: Optional[str]) -> Optional[str]:
        """Get Gstreamer encoder element name.

        :param encoder_type: Encoder type. Can be 'nvenc' or 'software'.
                             When not specified nv_encoder will be used if codec has it
                             or sw_encoder when codec does not have nv_encoder.
        """

        if not hasattr(self, '_nv_encoder'):
            if self.nv_encoder is not None and _check_element_exists(self.nv_encoder):
                self._nv_encoder = self.nv_encoder
            else:
                self._nv_encoder = None

        if not hasattr(self, '_sw_encoder'):
            if self.sw_encoder is not None and _check_element_exists(self.sw_encoder):
                self._sw_encoder = self.sw_encoder
            else:
                self._sw_encoder = None

        if encoder_type is None:
            return self._nv_encoder or self._sw_encoder

        if encoder_type == 'nvenc':
            if self._nv_encoder is None:
                raise ValueError(f'Codec {self.name} does not have nvenc encoder.')
            return self._nv_encoder

        if encoder_type == 'software':
            if self._sw_encoder is None:
                raise ValueError(f'Codec {self.name} does not have software encoder.')
            return self._sw_encoder

        raise ValueError(f'Unknown encoder type: {encoder_type}')


class Codec(Enum):
    """Codec enum."""

    H264 = CodecInfo(
        'h264',
        'video/x-h264',
        ['stream-format=byte-stream', 'alignment=au'],
        'h264parse',
        nv_encoder='nvv4l2h264enc',
        sw_encoder='x264enc',
    )
    HEVC = CodecInfo(
        'hevc',
        'video/x-h265',
        ['stream-format=byte-stream', 'alignment=au'],
        'h265parse',
        nv_encoder='nvv4l2h265enc',
    )
    RAW_RGBA = CodecInfo('raw-rgba', 'video/x-raw', ['format=RGBA'], is_raw=True)
    RAW_RGB24 = CodecInfo('raw-rgb24', 'video/x-raw', ['format=RGB'], is_raw=True)
    PNG = CodecInfo(
        'png',
        'image/png',
        [],
        'pngparse',
        sw_encoder='pngenc',
    )
    JPEG = CodecInfo(
        'jpeg',
        'image/jpeg',
        [],
        'jpegparse',
        nv_encoder='nvjpegenc',
        sw_encoder='jpegenc',
    )


CODEC_BY_NAME: Dict[str, Codec] = {x.value.name: x for x in Codec}
CODEC_BY_CAPS_NAME: Dict[str, Codec] = {x.value.caps_name: x for x in Codec}


def _check_element_exists(element_name: str):
    logger.debug('Check if element %r exists', element_name)
    elem_factory = Gst.ElementFactory.find(element_name)
    if elem_factory is not None:
        logger.debug('Found element %r', element_name)
        return True
    else:
        logger.debug('Element %r not found', element_name)
        return False
