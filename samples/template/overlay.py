"""Custom DrawFunc implementation."""
from savant.deepstream.drawfunc import NvDsDrawFunc
from savant.deepstream.meta.frame import NvDsFrameMeta
from savant.utils.artist import Artist


class Overlay(NvDsDrawFunc):
    """..."""

    def draw_on_frame(self, frame_meta: NvDsFrameMeta, artist: Artist):
        """"...

        :param frame_meta: Frame metadata.
        :param artist: Artist to draw on the frame.
        """
        super().draw_on_frame(frame_meta, artist)
