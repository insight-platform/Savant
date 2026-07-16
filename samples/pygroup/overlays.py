"""PyFuncs demonstrating the ``pygroup`` element.

Each pyfunc:
- draws something on the main frame
- creates and feeds its own auxiliary stream
- creates a nested telemetry span for the drawing section
"""

from abc import abstractmethod
from typing import Dict

import cv2

from savant.deepstream.auxiliary_stream import AuxiliaryStream
from savant.deepstream.meta.frame import NvDsFrameMeta
from savant.deepstream.opencv_utils import nvds_to_gpu_mat
from savant.deepstream.pyfunc import NvDsPyFuncPlugin
from savant.gstreamer import Gst
from savant.utils.artist import Artist, Position


class OverlayBase(NvDsPyFuncPlugin):
    """Common machinery for the pygroup demo overlays.

    Subclasses override :py:meth:`draw_overlay` to define what is drawn
    on the frame and set :py:attr:`DRAW_SPAN_NAME` for the telemetry span
    that wraps the drawing section.
    """

    DRAW_SPAN_NAME: str = 'draw'

    def __init__(
        self,
        aux_suffix: str,
        aux_width: int,
        aux_height: int,
        codec_params: Dict,
        line_color_rgba=(0, 255, 0, 255),
        **kwargs,
    ):
        self.aux_suffix = aux_suffix
        self.aux_width = aux_width
        self.aux_height = aux_height
        self.codec_params = codec_params
        self.line_color_rgba = tuple(line_color_rgba)
        # source_id -> AuxiliaryStream
        self.aux_streams: Dict[str, AuxiliaryStream] = {}
        super().__init__(**kwargs)

    def on_source_add(self, source_id: str):
        if source_id in self.aux_streams:
            return
        aux_source_id = f'{source_id}{self.aux_suffix}'
        self.logger.info(
            'Creating auxiliary stream %s for source %s.', aux_source_id, source_id
        )
        self.aux_streams[source_id] = self.auxiliary_stream(
            source_id=aux_source_id,
            width=self.aux_width,
            height=self.aux_height,
            codec_params=self.codec_params,
        )

    def on_source_eos(self, source_id: str):
        aux_stream = self.aux_streams.pop(source_id, None)
        if aux_stream is not None:
            aux_stream.eos()

    def on_stop(self) -> bool:
        self.aux_streams = {}
        return super().on_stop()

    def process_frame(self, buffer: Gst.Buffer, frame_meta: NvDsFrameMeta):
        stream = self.get_cuda_stream(frame_meta)

        with nvds_to_gpu_mat(buffer, frame_meta.frame_meta) as frame_mat:
            with frame_meta.telemetry_span.nested_span(self.DRAW_SPAN_NAME):
                with Artist(frame_mat, stream) as artist:
                    self.draw_overlay(artist, frame_meta)

            with frame_meta.telemetry_span.nested_span('aux-stream-publish'):
                aux_stream = self.aux_streams.get(frame_meta.source_id)
                if aux_stream is not None:
                    _, aux_buffer = aux_stream.create_frame(
                        pts=frame_meta.pts,
                        duration=frame_meta.duration,
                    )
                    with nvds_to_gpu_mat(aux_buffer, batch_id=0) as aux_mat:
                        cv2.cuda.resize(
                            src=frame_mat,
                            dst=aux_mat,
                            dsize=(self.aux_width, self.aux_height),
                            stream=stream,
                        )

    @abstractmethod
    def draw_overlay(self, artist: Artist, frame_meta: NvDsFrameMeta):
        """Draw the overlay specific to this pyfunc."""


class HorizontalLineOverlay(OverlayBase):
    """Draws a horizontal line and a "Step 1" label on the main frame."""

    DRAW_SPAN_NAME = 'draw-horizontal-line'

    def draw_overlay(self, artist: Artist, frame_meta: NvDsFrameMeta):
        width, height = artist.frame_wh
        mid_y = height // 2
        artist.add_line(
            pt1=(0, mid_y),
            pt2=(width - 1, mid_y),
            color=self.line_color_rgba,
            thickness=4,
        )
        artist.add_text(
            text=f'Step 1: horizontal line (frame {frame_meta.frame_num})',
            anchor=(20, 40),
            font_scale=0.8,
            font_thickness=2,
            font_color=(255, 255, 255, 255),
            bg_color=(0, 0, 0, 200),
            padding=(6, 6, 6, 6),
            anchor_point_type=Position.LEFT_TOP,
        )


class VerticalLineOverlay(OverlayBase):
    """Draws a vertical line and a "Step 2" label on the main frame
    (after Step 1, because pygroup executes sequentially).
    """

    DRAW_SPAN_NAME = 'draw-vertical-line'

    def draw_overlay(self, artist: Artist, frame_meta: NvDsFrameMeta):
        width, height = artist.frame_wh
        mid_x = width // 2
        artist.add_line(
            pt1=(mid_x, 0),
            pt2=(mid_x, height - 1),
            color=self.line_color_rgba,
            thickness=4,
        )
        artist.add_text(
            text=f'Step 2: vertical line (frame {frame_meta.frame_num})',
            anchor=(20, 90),
            font_scale=0.8,
            font_thickness=2,
            font_color=(255, 255, 255, 255),
            bg_color=(0, 0, 0, 200),
            padding=(6, 6, 6, 6),
            anchor_point_type=Position.LEFT_TOP,
        )
