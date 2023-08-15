from pygstsavantframemeta import gst_buffer_get_savant_frame_meta
from savant_rs.pipeline import VideoPipeline

from savant.gstreamer import Gst  # noqa:F401
from savant.utils.logging import get_logger

logger = get_logger(__name__)


def move_frame_as_is_pad_probe(
    pad: Gst.Pad,
    info: Gst.PadProbeInfo,
    video_pipeline: VideoPipeline,
    stage_from: str,
    stage_to: str,
) -> Gst.PadProbeReturn:
    buffer: Gst.Buffer = info.get_buffer()
    logger.debug(
        'Moving frame from %s to %s in buffer with PTS %s.',
        stage_from,
        stage_to,
        buffer.pts,
    )
    savant_frame_meta = gst_buffer_get_savant_frame_meta(buffer)
    # TODO: handle savant_frame_meta==None
    frame_id = savant_frame_meta.idx if savant_frame_meta else None
    logger.debug(
        'Moving frame from %s to %s in buffer with PTS %s. Frame ID: %s.',
        stage_from,
        stage_to,
        buffer.pts,
        frame_id,
    )
    video_pipeline.move_as_is(stage_from, stage_to, [frame_id])

    return Gst.PadProbeReturn.OK
