"""Overlay drawing the ROI and the scene-change status on the frame."""

from samples.scene_change.roi_injector import LABEL as ROI_LABEL
from samples.scene_change.scene_change import ATTR_CHANGED, ATTR_DISTANCE
from samples.scene_change.scene_change import ELEMENT_NAME as SCENE_CHANGE_ELEMENT_NAME
from savant.deepstream.drawfunc import NvDsDrawFunc
from savant.deepstream.meta.frame import NvDsFrameMeta
from savant.utils.artist import Artist, Position

COLOR_SCENE_STABLE = (64, 255, 32, 255)
COLOR_SCENE_CHANGED = (255, 64, 64, 255)
COLOR_NO_DATA = (255, 192, 0, 255)


class Overlay(NvDsDrawFunc):
    """Draws the ROI box, scene-change status, and distance."""

    def draw_on_frame(self, frame_meta: NvDsFrameMeta, artist: Artist):
        for obj_meta in frame_meta.objects:
            if obj_meta.label != ROI_LABEL:
                continue

            changed_attr = obj_meta.get_attr_meta(
                SCENE_CHANGE_ELEMENT_NAME, ATTR_CHANGED
            )
            dist_attr = obj_meta.get_attr_meta(SCENE_CHANGE_ELEMENT_NAME, ATTR_DISTANCE)
            changed = bool(changed_attr.value) if changed_attr is not None else False
            distance = dist_attr.value if dist_attr is not None else None

            if distance is None:
                text = 'No data'
                color = COLOR_NO_DATA
            else:
                if changed:
                    status = 'Scene changed'
                    color = COLOR_SCENE_CHANGED
                else:
                    status = 'Stable'
                    color = COLOR_SCENE_STABLE
                text = f'{status} | dist={distance:.3f}'

            artist.add_bbox(obj_meta.bbox, border_width=2, border_color=color)
            artist.add_text(
                text,
                (int(obj_meta.bbox.left), int(obj_meta.bbox.top)),
                padding=(2, 2, 2, 2),
                font_color=color,
                font_scale=0.8,
                anchor_point_type=Position.LEFT_TOP,
            )
