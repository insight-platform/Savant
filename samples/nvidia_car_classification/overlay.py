"""Draw func adds car classification models outputs: car color, car make, car type."""

from savant.deepstream.drawfunc import NvDsDrawFunc
from savant.deepstream.meta.frame import NvDsFrameMeta
from savant.utils.artist import Position, Artist


class Overlay(NvDsDrawFunc):
    def draw_on_frame(self, frame_meta: NvDsFrameMeta, artist: Artist):
        """DrawFunc's method where the drawing happens.
        Use artist's methods to add custom graphics to the frame.

        :param frame_meta: This frame's metadata.
        :artist: Artist object, provides high-level interface to drawing funcitons.
        """
        super().draw_on_frame(frame_meta, artist)
        for obj_meta in frame_meta.objects:
            attr_meta = obj_meta.get_attr_meta("Secondary_CarColor", 'car_color')
            if attr_meta is not None:
                artist.add_text(
                    str(attr_meta.value),
                    int(obj_meta.bbox.left),
                    int(obj_meta.bbox.top) + 20,
                    bg_color=(0, 0, 0, 255),
                    anchor_point=Position.LEFT_TOP,
                )
            attr_meta = obj_meta.get_attr_meta("Secondary_CarMake", 'car_make')
            if attr_meta is not None:
                artist.add_text(
                    str(attr_meta.value),
                    int(obj_meta.bbox.left),
                    int(obj_meta.bbox.top) + 38,
                    bg_color=(0, 0, 0, 255),
                    anchor_point=Position.LEFT_TOP,
                )
            attr_meta = obj_meta.get_attr_meta("Secondary_VehicleTypes", 'car_type')
            if attr_meta is not None:
                artist.add_text(
                    str(attr_meta.value),
                    int(obj_meta.bbox.left),
                    int(obj_meta.bbox.top) + 56,
                    bg_color=(0, 0, 0, 255),
                    anchor_point=Position.LEFT_TOP,
                )
