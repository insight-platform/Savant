from fractions import Fraction
from typing import NamedTuple

from savant_rs.primitives import VideoFrame

from savant.api.constants import DEFAULT_FRAMERATE
from savant.gstreamer import Gst
from savant.gstreamer.codecs import CODEC_BY_NAME, Codec


class FrameParams(NamedTuple):
    """Frame parameters."""

    codec: Codec
    width: str
    height: str
    framerate: str

    @staticmethod
    def from_video_frame(frame: VideoFrame):
        return FrameParams(
            codec=CODEC_BY_NAME[frame.codec],
            width=frame.width,
            height=frame.height,
            framerate=frame.framerate,
        )


def build_caps(params: FrameParams) -> Gst.Caps:
    """Caps factory."""
    try:
        framerate = Fraction(params.framerate)
    except (ZeroDivisionError, ValueError):
        framerate = Fraction(DEFAULT_FRAMERATE)
    framerate = Gst.Fraction(framerate.numerator, framerate.denominator)
    caps = Gst.Caps.from_string(params.codec.value.caps_with_params)
    caps.set_value('width', params.width)
    caps.set_value('height', params.height)
    caps.set_value('framerate', framerate)

    return caps
