from typing import Optional

import gi
import pyds
from savant_rs.pipeline import VideoPipeline

from . import pygstsavantframemeta

gi.require_version('Gst', '1.0')
from gi.repository import Gst


def gst_buffer_add_savant_batch_meta(
    buffer: Gst.Buffer,
    idx: int,
) -> Optional[pygstsavantframemeta.GstSavantBatchMeta]:
    """Add savant batch metadata to GStreamer buffer as GstMeta.

    :param buffer: GStreamer buffer.
    :param idx: Batch IDX.
    :return: Added metadata.
    """
    return pygstsavantframemeta.gst_buffer_add_savant_batch_meta(hash(buffer), idx)


def gst_buffer_get_savant_batch_meta(
    buffer: Gst.Buffer,
) -> Optional[pygstsavantframemeta.GstSavantBatchMeta]:
    """Get savant batch metadata to GStreamer buffer.

    :param buffer: GStreamer buffer.
    :return: Metadata.
    """
    return pygstsavantframemeta.gst_buffer_get_savant_batch_meta(hash(buffer))


def gst_buffer_add_nvds_savant_frame_meta(
    buffer: Gst.Buffer,
    idx: int,
) -> Optional[pygstsavantframemeta.GstSavantFrameMeta]:
    """Add savant frame metadata to GStreamer buffer as NvDsMeta.

    :param buffer: GStreamer buffer.
    :param idx: Frame IDX.
    :return: Added metadata.
    """
    return pygstsavantframemeta.gst_buffer_add_nvds_savant_frame_meta(hash(buffer), idx)


def gst_buffer_add_savant_frame_meta(
    buffer: Gst.Buffer,
    idx: int,
) -> Optional[pygstsavantframemeta.GstSavantFrameMeta]:
    """Add savant frame metadata to GStreamer buffer as GstMeta.

    :param buffer: GStreamer buffer.
    :param idx: Frame IDX.
    :return: Added metadata.
    """
    return pygstsavantframemeta.gst_buffer_add_savant_frame_meta(hash(buffer), idx)


def gst_buffer_get_nvds_savant_frame_meta(
    buffer: Gst.Buffer,
) -> Optional[pygstsavantframemeta.GstSavantFrameMeta]:
    """Get savant frame metadata from NvDsMeta of GStreamer buffer.

    :param buffer: GStreamer buffer.
    :return: Metadata.
    """
    return pygstsavantframemeta.gst_buffer_get_nvds_savant_frame_meta(hash(buffer))


def gst_buffer_get_savant_frame_meta(
    buffer: Gst.Buffer,
) -> Optional[pygstsavantframemeta.GstSavantFrameMeta]:
    """Get savant frame metadata to GStreamer buffer.

    :param buffer: GStreamer buffer.
    :return: Metadata.
    """
    return pygstsavantframemeta.gst_buffer_get_savant_frame_meta(hash(buffer))


def nvds_frame_meta_get_nvds_savant_frame_meta(
    frame_meta: pyds.NvDsFrameMeta,
) -> Optional[pygstsavantframemeta.GstSavantFrameMeta]:
    """Get savant frame metadata from NvDs frame metadata.

    :param frame_meta: NvDs frame metadata.
    :return: Metadata.
    """
    return pygstsavantframemeta.nvds_frame_meta_get_nvds_savant_frame_meta(frame_meta)


def add_convert_savant_frame_meta_pad_probe(pad: Gst.Pad, to_nvds: bool):
    """Add pad probe to convert savant frame metadata from NvDsMeta to GstMeta
    or vice versa.

    :param pad: GStreamer pad.
    :param to_nvds: Whether convert metadata from GstMeta to NvDsMeta or vice versa.
    """
    pygstsavantframemeta.add_convert_savant_frame_meta_pad_probe(hash(pad), to_nvds)


def add_move_frame_as_is_pad_probe(
    pad: Gst.Pad,
    video_pipeline: VideoPipeline,
    stage: str,
):
    pygstsavantframemeta.add_move_frame_as_is_pad_probe(
        hash(pad),
        video_pipeline.memory_handle,
        stage,
    )


def add_move_and_pack_frames_pad_probe(
    pad: Gst.Pad,
    video_pipeline: VideoPipeline,
    stage: str,
):
    pygstsavantframemeta.add_move_and_pack_frames_pad_probe(
        hash(pad),
        video_pipeline.memory_handle,
        stage,
    )


def add_move_batch_as_is_pad_probe(
    pad: Gst.Pad,
    video_pipeline: VideoPipeline,
    stage: str,
):
    pygstsavantframemeta.add_move_batch_as_is_pad_probe(
        hash(pad),
        video_pipeline.memory_handle,
        stage,
    )


def add_move_and_unpack_batch_pad_probe(
    pad: Gst.Pad,
    video_pipeline: VideoPipeline,
    stage: str,
):
    pygstsavantframemeta.add_move_and_unpack_batch_pad_probe(
        hash(pad),
        video_pipeline.memory_handle,
        stage,
    )
