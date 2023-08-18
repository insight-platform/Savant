from collections import defaultdict
from itertools import chain

import cv2

from samples.traffic_meter.utils import Direction, RandColorIterator
from savant.deepstream.drawfunc import NvDsDrawFunc
from savant.deepstream.meta.frame import BBox, NvDsFrameMeta
from savant.utils.artist import Artist, Position


class Overlay(NvDsDrawFunc):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.obj_colors = defaultdict(lambda: next(RandColorIterator()))

    def draw_on_frame(self, frame_meta: NvDsFrameMeta, artist: Artist):
        line_to = None
        line_from = None
        entries_n = None
        exits_n = None
        for obj_meta in frame_meta.objects:
            if obj_meta.is_primary:
                line_from = obj_meta.get_attr_meta('analytics', 'line_from')
                line_to = obj_meta.get_attr_meta('analytics', 'line_to')
                entries_n = obj_meta.get_attr_meta('analytics', 'entries_n')
                exits_n = obj_meta.get_attr_meta('analytics', 'exits_n')
            else:
                # mark obj center as it is used for entry/exit detection
                color = self.obj_colors[(frame_meta.source_id, obj_meta.track_id)]
                center = round(obj_meta.bbox.xc), round(obj_meta.bbox.yc)
                artist.add_circle(center, 3, color, cv2.FILLED)

                # add entry/exit label if detected
                entries = obj_meta.get_attr_meta_list(
                    'lc_tracker', Direction.entry.name
                )
                exits = obj_meta.get_attr_meta_list('lc_tracker', Direction.exit.name)
                entry_events_meta = entries if entries is not None else []
                exit_events_meta = exits if exits is not None else []
                offset = 20
                for attr_meta in chain(entry_events_meta, exit_events_meta):
                    direction = attr_meta.name
                    artist.add_text(
                        direction,
                        (int(obj_meta.bbox.left), int(obj_meta.bbox.top) + offset),
                        anchor_point_type=Position.LEFT_TOP,
                    )
                    offset += 20

        # draw boundary lines
        if line_from and line_to:
            pt1 = line_from.value[:2]
            pt2 = line_from.value[2:]
            artist.add_polygon([pt1, pt2], line_color=(255, 0, 0, 255))
            pt1 = line_to.value[:2]
            pt2 = line_to.value[2:]
            artist.add_polygon([pt1, pt2], line_color=(0, 0, 255, 255))

        # manually refresh (by filling with black) frame padding used for drawing
        # this workaround avoids rendering problem where drawings from previous frames
        # are persisted on the padding area in the next frame
        frame_w, _ = artist.frame_wh
        artist.add_bbox(
            BBox(
                frame_w // 2,
                self.overlay_height // 2,
                frame_w,
                self.overlay_height,
            ),
            border_width=0,
            bg_color=(0, 0, 0, 0),
        )
        # add entries/exits counters
        entries_n = entries_n.value if entries_n is not None else 0
        exits_n = exits_n.value if exits_n is not None else 0
        artist.add_text(
            f'Entries: {entries_n}',
            (50, 50),
            2.5,
            5,
            anchor_point_type=Position.LEFT_TOP,
        )
        artist.add_text(
            f'Exits: {exits_n}',
            (600, 50),
            2.5,
            5,
            anchor_point_type=Position.LEFT_TOP,
        )
