import cv2
import yaml

from savant.deepstream.drawfunc import NvDsDrawFunc
from savant.deepstream.meta.frame import BBox, NvDsFrameMeta
from savant.utils.artist import Artist, Position


class Overlay(NvDsDrawFunc):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        with open(self.config_path, 'r', encoding='utf8') as stream:
            self.area_config = yaml.safe_load(stream)

        self.areas = {}
        for source_id, areas in self.area_config.items():
            self.areas[source_id] = {}
            for area_name, area_dict in areas.items():
                self.areas[source_id][area_name] = (
                    area_dict['points'],
                    area_dict['color'],
                )

    def draw_on_frame(self, frame_meta: NvDsFrameMeta, artist: Artist):
        # manually refresh (by filling with black) frame padding used for drawing
        # this workaround avoids rendering problem where drawings from previous frames
        # are persisted on the padding area in the next frame
        frame_w, frame_h = artist.frame_wh
        artist.add_bbox(
            BBox(
                frame_w - self.sidebar_width // 2,
                frame_h // 2,
                self.sidebar_width,
                frame_h,
            ),
            border_width=0,
            bg_color=(0, 0, 0, 255),
        )

        if frame_meta.source_id not in self.areas:
            return

        primary_meta_object = None
        obj_metas = []
        for obj_meta in frame_meta.objects:
            if obj_meta.is_primary:
                primary_meta_object = obj_meta
            elif obj_meta.label == self.target_obj_label:
                obj_metas.append(obj_meta)

        if not primary_meta_object:
            return

        legend_rect_left = frame_w - self.sidebar_width + 75
        legend_rect_width = 50
        legend_rect_center_x = legend_rect_left + legend_rect_width // 2
        legend_text_x = legend_rect_left + legend_rect_width + 20
        legend_y = 50

        area_lines = self.areas[frame_meta.source_id]
        for area_name, (points, color) in area_lines.items():

            artist.add_polygon(points, line_color=color, line_width=2)

            for obj in obj_metas:
                if obj.draw_label == area_name:
                    center = round(obj.bbox.xc), round(obj.bbox.yc)
                    artist.add_circle(center, 3, color, cv2.FILLED)

            n_objs_meta = primary_meta_object.get_attr_meta('analytics', area_name)
            if n_objs_meta:
                n_objs = n_objs_meta.value
            else:
                n_objs = 0

            text_size, _ = artist.add_text(
                f'{n_objs:2d}',
                (legend_text_x, legend_y),
                2,
                4,
                anchor_point_type=Position.LEFT_TOP,
            )

            artist.add_bbox(
                BBox(
                    legend_rect_center_x,
                    legend_y + text_size[1] // 2,
                    legend_rect_width,
                    legend_rect_width,
                ),
                border_width=0,
                bg_color=color,
            )

            legend_y += legend_rect_width + 50
