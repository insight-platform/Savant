from collections import defaultdict
from itertools import chain

import cv2

from savant.deepstream.drawfunc import NvDsDrawFunc
from savant.deepstream.meta.frame import BBox, NvDsFrameMeta
from savant.utils.artist import Artist, Position

from ctypes import *
import pyds

STREAMMUX_WIDTH = 1920
STREAMMUX_HEIGHT = 1080

skeleton = [
    [16, 14],
    [14, 12],
    [17, 15],
    [15, 13],
    [12, 13],
    [6, 12],
    [7, 13],
    [6, 7],
    [6, 8],
    [7, 9],
    [8, 10],
    [9, 11],
    [2, 3],
    [1, 2],
    [1, 3],
    [2, 4],
    [3, 5],
    [4, 6],
    [5, 7],
]


class Overlay(NvDsDrawFunc):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # self.obj_colors = defaultdict(lambda: next(RandColorIterator()))

    def draw_on_frame(self, frame_meta: NvDsFrameMeta, artist: Artist):
        line_to = None
        line_from = None
        entries_n = None
        exits_n = None
        for obj in frame_meta.objects:
            if obj.label == "person":
                obj_meta = obj.object_meta_impl.ds_object_meta
                num_joints = int(obj_meta.mask_params.size / (sizeof(c_float) * 3))

                gain = min(
                    obj_meta.mask_params.width / STREAMMUX_WIDTH,
                    obj_meta.mask_params.height / STREAMMUX_HEIGHT,
                )
                pad_x = (obj_meta.mask_params.width - STREAMMUX_WIDTH * gain) / 2.0
                pad_y = (obj_meta.mask_params.height - STREAMMUX_HEIGHT * gain) / 2.0

                # batch_meta = frame_meta.batch_meta
                # display_meta = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
                # pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)
                key_points = []
                for i in range(num_joints):
                    data = obj_meta.mask_params.get_mask_array()
                    xc = int((data[i * 3 + 0] - pad_x) / gain)
                    yc = int((data[i * 3 + 1] - pad_y) / gain)
                    key_points.append((xc, yc))
                    confidence = data[i * 3 + 2]
                    # print("xc=", xc, "yc=", yc, "confidence=", confidence)
                    if xc > 0 and yc > 0:
                        artist.add_circle((xc, yc), 3, (255, 0, 0, 255), 3)
                for skelet_point in skeleton:
                    artist.add_polygon(
                        [
                            key_points[skelet_point[0] - 1],
                            key_points[skelet_point[1] - 1],
                        ],
                        line_color=(255, 0, 0, 255),
                        line_width=3,
                    )
        # for obj_meta in frame_meta.objects:
        #     if obj_meta.is_primary:
        #         line_from = obj_meta.get_attr_meta('analytics', 'line_from')
        #         line_to = obj_meta.get_attr_meta('analytics', 'line_to')
        #         entries_n = obj_meta.get_attr_meta('analytics', 'entries_n')
        #         exits_n = obj_meta.get_attr_meta('analytics', 'exits_n')
        #     else:
        #         # mark obj center as it is used for entry/exit detection
        #         color = self.obj_colors[(frame_meta.source_id, obj_meta.track_id)]
        #         center = round(obj_meta.bbox.xc), round(obj_meta.bbox.yc)
        #         artist.add_circle(center, 3, color, cv2.FILLED)
        #
        #         # add entry/exit label if detected
        #         entries = obj_meta.get_attr_meta_list(
        #             'lc_tracker', Direction.entry.name
        #         )
        #         exits = obj_meta.get_attr_meta_list('lc_tracker', Direction.exit.name)
        #         entry_events_meta = entries if entries is not None else []
        #         exit_events_meta = exits if exits is not None else []
        #         offset = 20
        #         for attr_meta in chain(entry_events_meta, exit_events_meta):
        #             direction = attr_meta.name
        #             artist.add_text(
        #                 direction,
        #                 (int(obj_meta.bbox.left), int(obj_meta.bbox.top) + offset),
        #                 anchor_point_type=Position.LEFT_TOP,
        #             )
        #             offset += 20
        #
        # # draw boundary lines
        # if line_from and line_to:
        #     pt1 = line_from.value[:2]
        #     pt2 = line_from.value[2:]
        #     artist.add_polygon([pt1, pt2], line_color=(255, 0, 0, 255))
        #     pt1 = line_to.value[:2]
        #     pt2 = line_to.value[2:]
        #     artist.add_polygon([pt1, pt2], line_color=(0, 0, 255, 255))
        #
        # # manually refresh (by filling with black) frame padding used for drawing
        # # this workaround avoids rendering problem where drawings from previous frames
        # # are persisted on the padding area in the next frame
        # frame_w, _ = artist.frame_wh
        # artist.add_bbox(
        #     BBox(
        #         frame_w // 2,
        #         self.overlay_height // 2,
        #         frame_w,
        #         self.overlay_height,
        #     ),
        #     border_width=0,
        #     bg_color=(0, 0, 0, 0),
        # )
        # # add entries/exits counters
        # entries_n = entries_n.value if entries_n is not None else 0
        # exits_n = exits_n.value if exits_n is not None else 0
        # artist.add_text(
        #     f'Entries: {entries_n}',
        #     (50, 50),
        #     2.5,
        #     5,
        #     anchor_point_type=Position.LEFT_TOP,
        # )
        # artist.add_text(
        #     f'Exits: {exits_n}',
        #     (600, 50),
        #     2.5,
        #     5,
        #     anchor_point_type=Position.LEFT_TOP,
        # )
