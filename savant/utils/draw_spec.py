from typing import Optional, Tuple
import copy
from savant_rs.draw_spec import (
    BoundingBoxDraw,
    ColorDraw,
    LabelDraw,
    DotDraw,
    PaddingDraw,
    ObjectDraw,
    LabelPosition,
    LabelPositionKind,
)


def convert_hex_to_rgba(hex_color: str) -> Tuple[int, int, int, int]:
    """Convert hex color to RGBA.
    Hex color string is expected to be exactly 8 characters long.

    :param hex_color: Hex color string
    :return: RGBA color tuple
    """
    return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4, 6))


def get_obj_draw_spec(config: Optional[dict]) -> ObjectDraw:
    """Create ObjectDraw from config."""
    if config is None:
        # config dict is None in case the user defined only the root of an object node
        # e.g.
        # peoplenet:
        #   person:
        return get_default_draw_spec(track_id=False)
    bbox_draw = None
    if 'bbox' in config:
        # default green borders
        border_color = config['bbox'].get('border_color', '00FF00FF')
        # default transparent background
        background_color = config['bbox'].get('background_color', '00000000')
        # default border thickness 2
        thickness = config['bbox'].get('thickness', 2)
        # default no padding

        if 'padding' in config['bbox']:
            padding_draw = PaddingDraw(**config['bbox']['padding'])
        else:
            padding_draw = PaddingDraw()
        bbox_draw = BoundingBoxDraw(
            border_color=ColorDraw(*convert_hex_to_rgba(border_color)),
            background_color=ColorDraw(*convert_hex_to_rgba(background_color)),
            padding=padding_draw,
            thickness=thickness,
        )

    central_dot_draw = None
    if 'central_dot' in config:
        # default green color
        color = config['central_dot'].get('color', '00FF00FF')
        # default radius 5
        radius = config['central_dot'].get('radius', 5)
        central_dot_draw = DotDraw(
            color=ColorDraw(*convert_hex_to_rgba(color)),
            radius=radius,
        )

    label_draw = None
    if 'label' in config:
        # default white font color
        font_color = config['label'].get('font_color', 'FFFFFFFF')
        # default transparent border
        border_color = config['label'].get('border_color', '00000000')
        # default black background
        background_color = config['label'].get('background_color', '000000FF')
        # default font scale 0.5
        font_scale = config['label'].get('font_scale', 0.5)
        # default font thickness 1
        thickness = config['label'].get('thickness', 1)
        # default format: {label}
        label_format = config['label'].get('format', ['{label}'])

        # rely on rust for defaults for label position
        if 'position' in config['label']:
            label_pos_kwargs = copy.deepcopy(config['label']['position'])
            if 'position' in label_pos_kwargs:
                if label_pos_kwargs['position'] == 'Center':
                    label_pos_kwargs['position'] = LabelPositionKind.Center
                elif label_pos_kwargs['position'] == 'TopLeftOutside':
                    label_pos_kwargs['position'] = LabelPositionKind.TopLeftOutside
                elif label_pos_kwargs['position'] == 'TopLeftInside':
                    label_pos_kwargs['position'] = LabelPositionKind.TopLeftInside
                else:
                    # invalid position kind
                    label_pos_kwargs.pop('position')
            label_position = LabelPosition(**label_pos_kwargs)
        else:
            label_position = LabelPosition()

        label_draw = LabelDraw(
            font_color=ColorDraw(*convert_hex_to_rgba(font_color)),
            border_color=ColorDraw(*convert_hex_to_rgba(border_color)),
            background_color=ColorDraw(*convert_hex_to_rgba(background_color)),
            font_scale=font_scale,
            thickness=thickness,
            format=label_format,
            position=label_position,
        )

    blur = config.get('blur', False)

    return ObjectDraw(
        bounding_box=bbox_draw,
        label=label_draw,
        central_dot=central_dot_draw,
        blur=blur,
    )


def get_default_draw_spec(track_id: bool = True) -> ObjectDraw:
    default_bbox_spec = BoundingBoxDraw(
        border_color=ColorDraw(red=0, green=255, blue=0, alpha=255),
    )
    default_bg_color = ColorDraw(red=0, green=0, blue=0, alpha=255)
    default_font_color = ColorDraw(red=255, green=255, blue=255, alpha=255)
    default_font_scale = 0.5
    if not track_id:
        return ObjectDraw(
            bounding_box=default_bbox_spec,
            label=LabelDraw(
                font_color=default_font_color,
                font_scale=default_font_scale,
                background_color=default_bg_color,
                format=['{label}'],
            ),
        )
    return ObjectDraw(
        bounding_box=default_bbox_spec,
        label=LabelDraw(
            font_color=default_font_color,
            font_scale=default_font_scale,
            background_color=default_bg_color,
            format=['{label} #{track_id}'],
        ),
    )
