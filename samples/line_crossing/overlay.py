
from itertools import chain
import cv2
from savant.deepstream.drawfunc import NvDsDrawFunc
from savant.deepstream.meta.frame import NvDsFrameMeta, BBox
from savant.utils.artist import Position, Artist
from samples.line_crossing.utils import Direction, RandColorIterator
from savant.meta.constants import UNTRACKED_OBJECT_ID

class Overlay(NvDsDrawFunc):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.rand_colors = RandColorIterator()
        self.obj_colors = {}

    def draw_on_frame(self, frame_meta: NvDsFrameMeta, artist: Artist):
        line_to = None
        line_from = None
        entries_n = None
        exits_n = None
        for obj_meta in frame_meta.objects:
            if obj_meta.is_primary:
                line_from = obj_meta.get_attr_meta(
                    'analytics', 'line_from'
                )
                line_to = obj_meta.get_attr_meta(
                    'analytics', 'line_to'
                )
                entries_n = obj_meta.get_attr_meta(
                    'analytics', 'entries_n'
                )
                exits_n = obj_meta.get_attr_meta(
                    'analytics', 'exits_n'
                )
            else:
                # mark obj center as it is used for entry/exit detection
                if (frame_meta.source_id, obj_meta.track_id) not in self.obj_colors:
                    self.obj_colors[(frame_meta.source_id, obj_meta.track_id)] = next(self.rand_colors)
                color = self.obj_colors[(frame_meta.source_id, obj_meta.track_id)]
                center = round(obj_meta.bbox.x_center), round(obj_meta.bbox.y_center)
                artist.add_circle(center, 3, color, cv2.FILLED)

                # default object visualisation
                # bbox + label + track id
                # artist.add_bbox(
                #     obj_meta.bbox,1,
                #     padding=0,
                # )
                # label = obj_meta.label
                # if obj_meta.track_id != UNTRACKED_OBJECT_ID:
                #     label += f' #{obj_meta.track_id}'
                # if isinstance(obj_meta.bbox, BBox):
                #     artist.add_text(
                #         text=label,
                #         anchor_x=int(obj_meta.bbox.left),
                #         anchor_y=int(obj_meta.bbox.top),
                #         bg_color=(0.0, 0.0, 0.0),
                #         anchor_point=Position.LEFT_TOP,
                #         padding=0
                #     )

                # add entry/exit label if detected
                entries = obj_meta.get_attr_meta_list(
                    'lc_tracker', Direction.entry.name
                )
                exits = obj_meta.get_attr_meta_list(
                    'lc_tracker', Direction.exit.name
                )
                entry_events_meta = entries if entries is not None else []
                exit_events_meta = exits if exits is not None else []
                offset = 20
                for attr_meta in chain(entry_events_meta, exit_events_meta):
                    direction = attr_meta.name
                    artist.add_text(
                        direction,
                        int(obj_meta.bbox.left),
                        int(obj_meta.bbox.top) + offset,
                        bg_color=(0, 0, 0),
                        padding=0,
                        anchor_point=Position.LEFT_TOP,
                    )
                    offset += 20

        # draw boundary lines
        if line_from and line_to:
            pt1 = line_from.value[:2]
            pt2 = line_from.value[2:]
            artist.add_polygon(
                [pt1, pt2],
                line_color=(0,0,1)
            )
            pt1 = line_to.value[:2]
            pt2 = line_to.value[2:]
            artist.add_polygon(
                [pt1, pt2],
                line_color=(1,0,0)
            )

        # manually refresh (by filling with black) frame padding used for drawing
        # this workaround avoids rendering problem where drawings from previous frames
        # are persisted on the padding area in the next frame
        frame_w, _ = artist.frame_wh
        artist.add_bbox(
            BBox(
                x_center=frame_w // 2,
                y_center=self.overlay_height // 2,
                width=frame_w,
                height=self.overlay_height,
            ),
            border_width=0,
            bg_color=(0, 0, 0),
            padding=0,
        )
        # add entries/exits counters
        entries_n = entries_n.value if entries_n is not None else 0
        exits_n = exits_n.value if exits_n is not None else 0
        artist.add_text(
            f'entries {entries_n}',
            50,
            50,
            2.5,
            5,
            (1, 1, 1),
            padding=0,
            anchor_point=Position.LEFT_TOP,
        )
        artist.add_text(
            f'exits {exits_n}',
            600,
            50,
            2.5,
            5,
            (1, 1, 1),
            padding=0,
            anchor_point=Position.LEFT_TOP,
        )
