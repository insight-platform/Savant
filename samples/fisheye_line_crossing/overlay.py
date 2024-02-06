from collections import defaultdict

import cv2
import yaml

from samples.intersection_traffic_meter.utils import RandColorIterator
from savant.deepstream.drawfunc import NvDsDrawFunc
from savant.deepstream.meta.frame import NvDsFrameMeta
from savant.utils.artist import Artist


class Overlay(NvDsDrawFunc):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.obj_colors = defaultdict(lambda: next(RandColorIterator()))

        with open(self.config_path, 'r', encoding='utf8') as stream:
            self.line_config = yaml.safe_load(stream)

        self.lines = {}
        for source_id, poly_cfg in self.line_config.items():
            self.lines[source_id] = {}
            src_lines = list(zip(poly_cfg['points'][:-1], poly_cfg['points'][1:]))
            src_lines.append((poly_cfg['points'][-1], poly_cfg['points'][0]))
            for line, edge_tag in zip(src_lines, poly_cfg['edges']):
                if edge_tag:
                    self.lines[source_id][edge_tag] = line

    def draw_on_frame(self, frame_meta: NvDsFrameMeta, artist: Artist):

        crossing_counts = defaultdict(int)
        for obj_meta in frame_meta.objects:
            if obj_meta.is_primary:
                for edge_tag in self.lines[frame_meta.source_id].keys():
                    dir_crossing_n = obj_meta.get_attr_meta('analytics', edge_tag)
                    crossing_counts[edge_tag] = (
                        dir_crossing_n.value if dir_crossing_n else 0
                    )

            else:
                # mark obj center and draw rotated bbox
                color = self.obj_colors[(frame_meta.source_id, obj_meta.track_id)]
                center = round(obj_meta.bbox.xc), round(obj_meta.bbox.yc)
                artist.add_circle(center, 3, color, cv2.FILLED)
                artist.add_bbox(obj_meta.bbox, 1)

        # draw boundary lines and crossings counts
        try:
            src_lines = self.lines[frame_meta.source_id]
        except KeyError:
            pass
        else:
            for edge_tag, line in src_lines.items():
                artist.add_polygon(line, line_color=(255, 0, 0, 255), line_width=2)
                crossing_n = crossing_counts[edge_tag]
                artist.add_text(f'{crossing_n}', line[0])
