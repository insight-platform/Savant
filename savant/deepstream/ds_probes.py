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
    stage: str,
) -> Gst.PadProbeReturn:
    buffer: Gst.Buffer = info.get_buffer()
    logger.debug('Pipeline stage %s. Packing frames at buffer %s.', stage, buffer.pts)
    frames_ids = []
    nvds_batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(buffer))
    for nvds_frame_meta in nvds_frame_meta_iterator(nvds_batch_meta):
        savant_frame_meta = nvds_frame_meta_get_nvds_savant_frame_meta(nvds_frame_meta)
        if savant_frame_meta is None:
            logger.warning(
                'Pipeline stage %s. Failed to pack frames at buffer %s to batch. '
                'Frame %s has no Savant Frame Meta.',
                stage,
                buffer.pts,
                nvds_frame_meta.buf_pts,
            )
            return Gst.PadProbeReturn.PASS
        frames_ids.append(savant_frame_meta.idx if savant_frame_meta else None)

    if not frames_ids:
        logger.debug(
            'Pipeline stage %s. Skipping buffer %s: batch is empty.',
            stage,
            buffer.pts,
        )
        return Gst.PadProbeReturn.PASS

    batch_id = video_pipeline.move_and_pack_frames(stage, frames_ids, no_gil=False)
    logger.debug(
        'Pipeline stage %s. Frames %s at buffer %s packed to batch %s.',
        stage,
        frames_ids,
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
    stage: str,
) -> Gst.PadProbeReturn:
    buffer: Gst.Buffer = info.get_buffer()
    logger.debug('Pipeline stage %s. Moving batch at buffer %s.', stage, buffer.pts)
    nvds_batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(buffer))
    try:
        next(nvds_frame_meta_iterator(nvds_batch_meta))
    except StopIteration:
        logger.debug(
            'Pipeline stage %s. Skipping buffer %s: batch is empty.',
            stage,
            buffer.pts,
        )
        return Gst.PadProbeReturn.PASS

    savant_frame_meta = gst_buffer_get_savant_frame_meta(buffer)
    if savant_frame_meta is None:
        logger.warning(
            'Pipeline stage %s. Failed to move batch at buffer %s. '
            'Batch has no Savant Frame Meta.',
            stage,
            buffer.pts,
        )
        return Gst.PadProbeReturn.PASS

    batch_id = savant_frame_meta.idx if savant_frame_meta else None
    logger.debug(
        'Moving batch to %s in buffer with PTS %s. Batch ID: %s.',
        stage,
        buffer.pts,
        batch_id,
    )
    video_pipeline.move_as_is(stage, [batch_id], no_gil=False)
    logger.debug(
        'Pipeline stage %s. Batch %s at buffer %s moved.',
        stage,
        batch_id,
        buffer.pts,
    )

    return Gst.PadProbeReturn.OK


def move_batch_to_frames_pad_probe(
    pad: Gst.Pad,
    info: Gst.PadProbeInfo,
    video_pipeline: VideoPipeline,
    stage: str,
) -> Gst.PadProbeReturn:
    buffer: Gst.Buffer = info.get_buffer()
    logger.debug('Pipeline stage %s. Unpacking batch at buffer %s.', stage, buffer.pts)
    nvds_batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(buffer))
    try:
        next(nvds_frame_meta_iterator(nvds_batch_meta))
    except StopIteration:
        logger.debug(
            'Pipeline stage %s. Skipping buffer %s: batch is empty.',
            stage,
            buffer.pts,
        )
        return Gst.PadProbeReturn.PASS

    savant_frame_meta = gst_buffer_get_savant_frame_meta(buffer)
    if savant_frame_meta is None:
        logger.warning(
            'Pipeline stage %s. Failed to unpack batch at buffer %s. '
            'Batch has no Savant Frame Meta.',
            stage,
            buffer.pts,
        )
        return Gst.PadProbeReturn.PASS

    batch_id = savant_frame_meta.idx if savant_frame_meta else None
    frame_map: Dict[str, int] = video_pipeline.move_and_unpack_batch(
        stage,
        batch_id,
        no_gil=False,
    )
    logger.debug(
        'Pipeline stage %s. Batch %s at buffer %s unpacked to frames %s.',
        stage,
        batch_id,
        buffer.pts,
        frame_map,
    )

    return Gst.PadProbeReturn.OK


def add_move_frames_to_batch_pad_probe(
    pad: Gst.Pad,
    video_pipeline: VideoPipeline,
    stage: str,
):
    pad.add_probe(
        Gst.PadProbeType.BUFFER,
        move_frames_to_batch_pad_probe,
        video_pipeline,
        stage,
    )


def add_move_batch_as_is_pad_probe(
    pad: Gst.Pad,
    video_pipeline: VideoPipeline,
    stage: str,
):
    pad.add_probe(
        Gst.PadProbeType.BUFFER,
        move_batch_as_is_pad_probe,
        video_pipeline,
        stage,
    )


def add_move_batch_to_frames_pad_probe(
    pad: Gst.Pad,
    video_pipeline: VideoPipeline,
    stage: str,
):
    pad.add_probe(
        Gst.PadProbeType.BUFFER,
        move_batch_to_frames_pad_probe,
        video_pipeline,
        stage,
    )
