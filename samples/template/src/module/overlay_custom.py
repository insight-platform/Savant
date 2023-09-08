"""Custom DrawFunc implementation."""
from savant.deepstream.drawfunc import NvDsDrawFunc
from savant.deepstream.meta.frame import NvDsFrameMeta
from savant.utils.artist import Artist


class Overlay(NvDsDrawFunc):
    """Custom implementation of PyFunc for drawing on frame."""

    def draw_on_frame(self, frame_meta: NvDsFrameMeta, artist: Artist):
        """Draws on frame using the artist and the frame's metadata.

        :param frame_meta: Frame metadata.
        :param artist: Artist to draw on the frame.
        """
        # When the dev_mode is enabled in the module config
        # The draw func code changes are applied without restarting the module

        # super().draw_on_frame(frame_meta, artist)

        # for example, draw a white bounding box around persons
        # and a green bounding box around faces
        for obj in frame_meta.objects:
            if obj.label == 'person':
                artist.add_bbox(obj.bbox, 3, (255, 255, 255, 255))
            elif obj.label == 'face':
                artist.add_bbox(obj.bbox, 3, (0, 255, 0, 255))
