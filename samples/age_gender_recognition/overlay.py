"""Overlay dashboard module."""
from pathlib import Path
import cv2
from savant_rs.draw_spec import ObjectDraw, LabelDraw

from savant.deepstream.drawfunc import NvDsDrawFunc
from savant.deepstream.meta.frame import NvDsFrameMeta, BBox
from savant.meta.object import ObjectMeta
from savant.utils.artist import Position, Artist
from samples.peoplenet_detector.animation import Animation
from samples.peoplenet_detector.utils import load_sprite, get_font_scale


class Overlay(NvDsDrawFunc):

    def override_draw_spec(
        self, object_meta: ObjectMeta, draw_spec: ObjectDraw
    ) -> ObjectDraw:
        """Override draw spec for object with label 'Face'.
        Add age and gender attributes to the object visualisation.
        """
        if object_meta.label == 'face':
            new_label_format = draw_spec.label.format

            # one attribute per line
            # if there's no specific attribute for the object on this frame
            # reserve a line for it anyway
            # so that the object's labels don't jump up and down
            attr_meta = object_meta.get_attr_meta("smoothed_value", 'age')
            if attr_meta is not None:
                new_label_format += [f"age: {round(attr_meta.value)}"]
            else:
                new_label_format += ['']

            attr_meta = object_meta.get_attr_meta("smoothed_value", 'gender')
            if attr_meta is not None:
                new_label_format += [f"gender: {str(attr_meta.value)}"]
            else:
                new_label_format += ['']

            # draw_spec.label.format = new_label_format
            draw_spec = ObjectDraw(
                bounding_box=draw_spec.bounding_box,
                label=LabelDraw(
                    font_color=draw_spec.label.font_color,
                    border_color=draw_spec.label.border_color,
                    background_color=draw_spec.label.background_color,
                    padding=draw_spec.label.padding,
                    font_scale=draw_spec.label.font_scale,
                    thickness=draw_spec.label.thickness,
                    format=new_label_format,
                    position=draw_spec.label.position,
                ),
                central_dot=draw_spec.central_dot,
                blur=draw_spec.blur,
            )
        return draw_spec

    # def draw_on_frame(self, frame_meta: NvDsFrameMeta, artist: Artist):
    #     super().draw_on_frame(frame_meta, artist)
    #     for i, obj_meta in enumerate(frame_meta.objects):
    #         if not obj_meta.is_primary:
    #             # mark obj center as it is used for entry/exit detection
    #             color = (0, 255, 0, 255)
    #             landmarks = obj_meta.get_attr_meta(
    #                 'yolov5face', 'landmarks'
    #             ).value
    #
    #             for x, y in zip(landmarks[::2], landmarks[1::2]):
    #                 artist.add_circle((round(x), round(y)), 2, color, cv2.FILLED)
    #             # age = obj_meta.get_attr_meta(
    #             #     'age_gender', 'age'
    #             # ).value
    #             # gender = obj_meta.get_attr_meta(
    #             #     'age_gender', 'gender'
    #             # ).value
    #             # artist.add_text(f"{round(age)} - {gender[0]}", anchor=(round(landmarks[4]), round(landmarks[5])))
    #             # spec = self.default_spec_no_track_id
    #             # self._draw_bounding_box(obj_meta, artist, spec.bounding_box)
    #             # center = round(obj_meta.bbox.x_center), round(obj_meta.bbox.y_center)
    #             # artist.add_circle(center, 3, color, cv2.FILLED)
    #             #
    #             # # add entry/exit label if detected
    #             # entries = obj_meta.get_attr_meta_list(
    #             #     'lc_tracker', Direction.entry.name
    #             # )
    #             # exits = obj_meta.get_attr_meta_list('lc_tracker', Direction.exit.name)
    #             # entry_events_meta = entries if entries is not None else []
    #             # exit_events_meta = exits if exits is not None else []
    #             # offset = 20
    #             # for attr_meta in chain(entry_events_meta, exit_events_meta):
    #             #     direction = attr_meta.name
    #             #     artist.add_text(
    #             #         direction,
    #             #         (int(obj_meta.bbox.left), int(obj_meta.bbox.top) + offset),
    #             #         anchor_point_type=Position.LEFT_TOP,
    #             #     )
    #             #     offset += 20
