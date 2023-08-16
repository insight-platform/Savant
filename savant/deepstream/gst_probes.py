from pygstsavantframemeta import gst_buffer_get_savant_frame_meta
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
    buffer: Gst.Buffer = info.get_buffer()
    logger.debug('Moving frame to %s in buffer with PTS %s.', stage, buffer.pts)
    savant_frame_meta = gst_buffer_get_savant_frame_meta(buffer)
    # TODO: handle savant_frame_meta==None
    frame_id = savant_frame_meta.idx if savant_frame_meta else None
    logger.debug(
        'Moving frame to %s in buffer with PTS %s. Frame ID: %s.',
        stage,
        buffer.pts,
        frame_id,
    )
    video_pipeline.move_as_is(stage, [frame_id], no_gil=False)

    return Gst.PadProbeReturn.OK


def add_move_frame_as_is_pad_probe(
    pad: Gst.Pad,
    video_pipeline: VideoPipeline,
    stage: str,
):
    pad.add_probe(
        Gst.PadProbeType.BUFFER,
        move_frame_as_is_pad_probe,
        video_pipeline,
        stage,
    )
