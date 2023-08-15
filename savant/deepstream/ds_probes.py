from typing import Dict

import pyds
from pygstsavantframemeta import (
    gst_buffer_add_savant_frame_meta,
    gst_buffer_get_savant_frame_meta,
    nvds_frame_meta_get_nvds_savant_frame_meta,
)
from savant_rs.pipeline import VideoPipeline

from savant.deepstream.utils import nvds_frame_meta_iterator
from savant.gstreamer import Gst  # noqa:F401
from savant.utils.logging import get_logger

logger = get_logger(__name__)


def move_frames_to_batch_pad_probe(
    pad: Gst.Pad,
    info: Gst.PadProbeInfo,
    video_pipeline: VideoPipeline,
    stage_from: str,
    stage_to: str,
) -> Gst.PadProbeReturn:
    buffer: Gst.Buffer = info.get_buffer()
    logger.debug(
        'Moving frames to batch from %s to %s in buffer with PTS %s.',
        stage_from,
        stage_to,
        buffer.pts,
    )
    frames_ids = []
    nvds_batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(buffer))
    for nvds_frame_meta in nvds_frame_meta_iterator(nvds_batch_meta):
        # TODO: handle savant_frame_meta==None
        savant_frame_meta = nvds_frame_meta_get_nvds_savant_frame_meta(nvds_frame_meta)
        frames_ids.append(savant_frame_meta.idx if savant_frame_meta else None)
    if not frames_ids:
        logger.debug(
            'Moving frames to batch from %s to %s in buffer with PTS %s. Batch is empty.',
            stage_from,
            stage_to,
            buffer.pts,
        )
        return Gst.PadProbeReturn.OK

    logger.debug(
        'Moving frames to batch from %s to %s in buffer with PTS %s. Frames IDs: %s.',
        stage_from,
        stage_to,
        buffer.pts,
        frames_ids,
    )
    batch_id = video_pipeline.move_and_pack_frames(
        stage_from,
        stage_to,
        frames_ids,
    )
    logger.debug(
        'Moving frames to batch from %s to %s in buffer with PTS %s. Batch ID: %s.',
        stage_from,
        stage_to,
        buffer.pts,
        batch_id,
    )
    savant_frame_meta = gst_buffer_get_savant_frame_meta(buffer)
    if savant_frame_meta is not None:
        savant_frame_meta.idx = batch_id
    else:
        gst_buffer_add_savant_frame_meta(buffer, batch_id)

    return Gst.PadProbeReturn.OK


def move_batch_as_is_pad_probe(
    pad: Gst.Pad,
    info: Gst.PadProbeInfo,
    video_pipeline: VideoPipeline,
    stage_from: str,
    stage_to: str,
) -> Gst.PadProbeReturn:
    buffer: Gst.Buffer = info.get_buffer()
    logger.debug(
        'Moving batch from %s to %s in buffer with PTS %s.',
        stage_from,
        stage_to,
        buffer.pts,
    )
    nvds_batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(buffer))
    try:
        next(nvds_frame_meta_iterator(nvds_batch_meta))
    except StopIteration:
        logger.debug(
            'Moving batch from %s to %s in buffer with PTS %s. Batch is empty.',
            stage_from,
            stage_to,
            buffer.pts,
        )
        return Gst.PadProbeReturn.OK

    savant_frame_meta = gst_buffer_get_savant_frame_meta(buffer)
    # TODO: handle savant_frame_meta==None
    batch_id = savant_frame_meta.idx if savant_frame_meta else None
    logger.debug(
        'Moving batch from %s to %s in buffer with PTS %s. Batch ID: %s.',
        stage_from,
        stage_to,
        buffer.pts,
        batch_id,
    )
    video_pipeline.move_as_is(stage_from, stage_to, [batch_id])

    return Gst.PadProbeReturn.OK


def move_batch_to_frames_pad_probe(
    pad: Gst.Pad,
    info: Gst.PadProbeInfo,
    video_pipeline: VideoPipeline,
    stage_from: str,
    stage_to: str,
) -> Gst.PadProbeReturn:
    buffer: Gst.Buffer = info.get_buffer()
    logger.debug(
        'Moving batch to frames from %s to %s in buffer with PTS %s.',
        stage_from,
        stage_to,
        buffer.pts,
    )
    nvds_batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(buffer))
    try:
        next(nvds_frame_meta_iterator(nvds_batch_meta))
    except StopIteration:
        logger.debug(
            'Moving batch to frames from %s to %s in buffer with PTS %s. Batch is empty.',
            stage_from,
            stage_to,
            buffer.pts,
        )
        return Gst.PadProbeReturn.OK

    savant_frame_meta = gst_buffer_get_savant_frame_meta(buffer)
    # TODO: handle savant_frame_meta==None
    batch_id = savant_frame_meta.idx if savant_frame_meta else None
    logger.debug(
        'Moving batch to frames from %s to %s in buffer with PTS %s. Batch ID: %s.',
        stage_from,
        stage_to,
        buffer.pts,
        batch_id,
    )
    frame_map: Dict[str, int] = video_pipeline.move_and_unpack_batch(
        stage_from,
        stage_to,
        batch_id,
    )
    logger.debug(
        'Moving batch to frames from %s to %s in buffer with PTS %s. Frame map: %s.',
        stage_from,
        stage_to,
        buffer.pts,
        frame_map,
    )

    return Gst.PadProbeReturn.OK
