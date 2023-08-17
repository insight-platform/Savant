from pygstsavantframemeta import pygstsavantframemeta
from savant_rs.pipeline import VideoPipeline

from savant.gstreamer import Gst  # noqa:F401
from savant.utils.logging import get_logger

logger = get_logger(__name__)


def move_frame_as_is_pad_probe(
    pad: Gst.Pad,
    info: Gst.PadProbeInfo,
    video_pipeline: VideoPipeline,
    stage: str,
) -> Gst.PadProbeReturn:
    pygstsavantframemeta.move_frame_as_is_pad_probe(
        hash(pad),
        hash(info),
        video_pipeline.memory_handle,
        stage,
    )

    return Gst.PadProbeReturn.OK


def add_move_frame_as_is_pad_probe(
    pad: Gst.Pad,
    video_pipeline: VideoPipeline,
    stage: str,
):
    logger.debug('Pipeline stage %s. Adding move frame as is pad probe.', stage)
    return pad.add_probe(
        Gst.PadProbeType.BUFFER,
        move_frame_as_is_pad_probe,
        video_pipeline,
        stage,
    )
