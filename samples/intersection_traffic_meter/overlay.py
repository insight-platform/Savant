from collections import defaultdict
from itertools import combinations
import yaml
import cv2

from samples.intersection_traffic_meter.utils import RandColorIterator
from savant.deepstream.drawfunc import NvDsDrawFunc
from savant.deepstream.meta.frame import NvDsFrameMeta
from savant.utils.artist import Artist, Position


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

        self.directions = {}
        for source_id, src_lines in self.lines.items():

            src_edge_tags = list(src_lines.keys())

            src_directions = []
            # simple entries/exits
            for edge_tag in src_edge_tags:
                src_directions.append('+'.join(['entry', edge_tag]))
                src_directions.append('+'.join(['exit', edge_tag]))
            # full crossings
            for entry_tag, exit_tag in combinations(src_edge_tags, 2):
                src_directions.append('->'.join([entry_tag, exit_tag]))
                src_directions.append('->'.join([exit_tag, entry_tag]))

            self.directions[source_id] = src_directions


    def draw_on_frame(self, frame_meta: NvDsFrameMeta, artist: Artist):

        crossing_counts = defaultdict(int)
        for obj_meta in frame_meta.objects:
            if obj_meta.is_primary:
                for direction in self.directions[frame_meta.source_id]:
                    dir_crossing_n = obj_meta.get_attr_meta(
                        'analytics', direction
                    )
                    crossing_counts[direction] = dir_crossing_n.value if dir_crossing_n else 0

            else:
                # mark obj center
                color = self.obj_colors[(frame_meta.source_id, obj_meta.track_id)]
                center = round(obj_meta.bbox.xc), round(obj_meta.bbox.yc)
                artist.add_circle(center, 3, color, cv2.FILLED)

                # add crossing direction label if present
                all_events_meta = []
                for direction in self.directions[frame_meta.source_id]:
                    dir_events_meta = obj_meta.get_attr_meta_list(
                        'lc_tracker', direction
                    )
                    dir_events_meta = dir_events_meta if dir_events_meta else []
                    all_events_meta.extend(dir_events_meta)

                offset = 20
                for attr_meta in all_events_meta:
                    direction = attr_meta.name
                    artist.add_text(
                        direction,
                        (int(obj_meta.bbox.left), int(obj_meta.bbox.top) + offset),
                        anchor_point_type=Position.LEFT_TOP,
                    )
                    offset += 20

        # draw boundary lines and exit counts
        try:
            src_lines = self.lines[frame_meta.source_id]
        except KeyError:
            pass
        else:
            for edge_tag, line in src_lines.items():
                artist.add_polygon(line, line_color=(255, 0, 0, 255), line_width=2)
                edge_entry_n = 0
                edge_exit_n = 0
                for crossing_name, crossing_n in crossing_counts.items():
                    if '+' in crossing_name:
                        direction, edge = crossing_name.split('+')
                        if edge == edge_tag:
                            if 'entry' in direction:
                                edge_entry_n += crossing_n
                            else:
                                edge_exit_n += crossing_n
                    elif '->' in crossing_name:
                        _, exit_edge = crossing_name.split('->')
                        if exit_edge == edge_tag:
                            edge_exit_n += crossing_n

                artist.add_text(f'In: {edge_entry_n}, Out: {edge_exit_n}', line[0])
