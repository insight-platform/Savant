from pygstsavantframemeta import pygstsavantframemeta
from savant_rs.pipeline import VideoPipeline

from savant.gstreamer import Gst  # noqa:F401
from savant.utils.logging import get_logger

logger = get_logger(__name__)


def move_and_pack_frames_pad_probe(
    pad: Gst.Pad,
    info: Gst.PadProbeInfo,
    video_pipeline: VideoPipeline,
    stage: str,
) -> Gst.PadProbeReturn:
    pygstsavantframemeta.move_and_pack_frames_pad_probe(
        hash(pad),
        hash(info),
        video_pipeline.memory_handle,
        stage,
    )

    return Gst.PadProbeReturn.OK


def move_batch_as_is_pad_probe(
    pad: Gst.Pad,
    info: Gst.PadProbeInfo,
    video_pipeline: VideoPipeline,
    stage: str,
) -> Gst.PadProbeReturn:
    pygstsavantframemeta.move_batch_as_is_pad_probe(
        hash(pad),
        hash(info),
        video_pipeline.memory_handle,
        stage,
    )

    return Gst.PadProbeReturn.OK


def pipeline_move_and_unpack_batch(
    pad: Gst.Pad,
    info: Gst.PadProbeInfo,
    video_pipeline: VideoPipeline,
    stage: str,
) -> Gst.PadProbeReturn:
    pygstsavantframemeta.move_and_unpack_batch_pad_probe(
        hash(pad),
        hash(info),
        video_pipeline.memory_handle,
        stage,
    )

    return Gst.PadProbeReturn.OK


def add_move_and_pack_frames_pad_probe(
    pad: Gst.Pad,
    video_pipeline: VideoPipeline,
    stage: str,
):
    logger.debug('Pipeline stage %s. Adding move and pack frame pad probe.', stage)
    return pad.add_probe(
        Gst.PadProbeType.BUFFER,
        move_and_pack_frames_pad_probe,
        video_pipeline,
        stage,
    )


def add_move_batch_as_is_pad_probe(
    pad: Gst.Pad,
    video_pipeline: VideoPipeline,
    stage: str,
):
    logger.debug('Pipeline stage %s. Adding move batch as is pad probe.', stage)
    return pad.add_probe(
        Gst.PadProbeType.BUFFER,
        move_batch_as_is_pad_probe,
        video_pipeline,
        stage,
    )


def add_pipeline_move_and_unpack_batch(
    pad: Gst.Pad,
    video_pipeline: VideoPipeline,
    stage: str,
):
    logger.debug('Pipeline stage %s. Adding move and unpack batch pad probe.', stage)
    return pad.add_probe(
        Gst.PadProbeType.BUFFER,
        pipeline_move_and_unpack_batch,
        video_pipeline,
        stage,
    )
