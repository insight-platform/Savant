from collections import defaultdict
from itertools import combinations

import cv2
import yaml

from savant.deepstream.drawfunc import NvDsDrawFunc
from savant.deepstream.meta.frame import  BBox,NvDsFrameMeta
from savant.utils.artist import Artist, Position


class Overlay(NvDsDrawFunc):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


        with open(self.config_path, 'r', encoding='utf8') as stream:
            self.area_config = yaml.safe_load(stream)

        self.lines = {}
        for source_id, areas in self.area_config.items():
            self.lines[source_id] = {}
            for area_name, coords_list in areas.items():

                src_lines = list(zip(coords_list[:-1], coords_list[1:]))
                src_lines.append((coords_list[-1], coords_list[0]))

                self.lines[source_id][area_name] = src_lines

        self.logger.info(self.lines)



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

        if frame_meta.source_id not in self.lines:
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


        area_lines = self.lines[frame_meta.source_id]
        for area_name, lines in area_lines.items():

            for line in lines:
                artist.add_polygon(line, line_color=(255, 0, 0, 255), line_width=2)

            for obj in obj_metas:
                if obj.draw_label == area_name:
                    center = round(obj.bbox.xc), round(obj.bbox.yc)
                    artist.add_circle(center, 3, (255,0,255,255), cv2.FILLED)

            n_objs = primary_meta_object.get_attr_meta('analytics', area_name)
            if n_objs:
                text_anchor_x = frame_w - self.sidebar_width + 20
                text_anchor_y = 50
                artist.add_text(
                    f'{n_objs.value}',
                    (text_anchor_x, text_anchor_y),
                    2.5,
                    5,
                    anchor_point_type=Position.LEFT_TOP,
                )
