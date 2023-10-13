"""Draw func adds license plate"""
from savant_rs.draw_spec import LabelDraw, ObjectDraw

from savant.deepstream.drawfunc import NvDsDrawFunc
from savant.meta.object import ObjectMeta


class Overlay(NvDsDrawFunc):
    def override_draw_spec(
        self, object_meta: ObjectMeta, draw_spec: ObjectDraw
    ) -> ObjectDraw:
        """Override draw spec for objects with label 'lpd' (license plate detection).
        Add classifier attributes labels to the object visualisation.
        """
        if object_meta.label == 'lpd':

            # drawing a licence plate instead of the label or empty label
            attr_meta = object_meta.get_attr_meta('LPRNet', 'lpr')
            if attr_meta is not None:
                new_label_format = [str(attr_meta.value)]
            else:
                new_label_format = ['']

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
