"""Draw func adds car classification models outputs: car color, car make, car type."""
from savant_rs.draw_spec import LabelDraw, ObjectDraw

from savant.deepstream.drawfunc import NvDsDrawFunc
from savant.meta.object import ObjectMeta


class Overlay(NvDsDrawFunc):
    def override_draw_spec(
        self, object_meta: ObjectMeta, draw_spec: ObjectDraw
    ) -> ObjectDraw:
        """Override draw spec for objects with label 'Car'.
        Add classifier attributes labels to the object visualisation.
        """
        if object_meta.label == 'Car':
            new_label_format = draw_spec.label.format

            # one attribute per line
            # if there's no specific attribute for the object on this frame
            # reserve a line for it anyway
            # so that the object's labels don't jump up and down
            attr_meta = object_meta.get_attr_meta("Secondary_CarColor", 'car_color')
            if attr_meta is not None:
                new_label_format += [str(attr_meta.value)]
            else:
                new_label_format += ['']

            attr_meta = object_meta.get_attr_meta("Secondary_CarMake", 'car_make')
            if attr_meta is not None:
                new_label_format += [str(attr_meta.value)]
            else:
                new_label_format += ['']

            attr_meta = object_meta.get_attr_meta("Secondary_VehicleTypes", 'car_type')
            if attr_meta is not None:
                new_label_format += [str(attr_meta.value)]
            else:
                new_label_format += ['']

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
