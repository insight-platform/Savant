"""Custom DrawFunc implementation."""
from savant_rs.draw_spec import ObjectDraw, BoundingBoxDraw, ColorDraw, PaddingDraw
from savant.deepstream.drawfunc import NvDsDrawFunc
from savant.meta.object import ObjectMeta


class Overlay(NvDsDrawFunc):
    """Custom implementation of PyFunc for drawing on frame."""

    def override_draw_spec(
        self, object_meta: ObjectMeta, draw_spec: ObjectDraw
    ) -> ObjectDraw:
        """Override draw spec for objects.
        """
        # When the dev_mode is enabled in the module config
        # The draw func code changes are applied without restarting the module

        if object_meta.label == 'person':
            # For example, change the border color of the bounding box
            # by specifying the new RGBA color in the draw spec
            bbox_draw = BoundingBoxDraw(
                border_color=ColorDraw(255, 255, 255, 255),
                background_color=ColorDraw(0, 0, 0, 0),
                thickness=1,
                padding=PaddingDraw(),
            )

            draw_spec = ObjectDraw(
                bounding_box=bbox_draw,
                label=draw_spec.label,
                central_dot=draw_spec.central_dot,
                blur=draw_spec.blur,
            )

        elif object_meta.label == 'face':
            # For example, switch face blur on or off
            draw_spec = ObjectDraw(
                bounding_box=draw_spec.bounding_box,
                label=draw_spec.label,
                central_dot=draw_spec.central_dot,
                blur=True,
            )
        return draw_spec
