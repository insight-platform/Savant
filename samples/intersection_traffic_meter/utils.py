"""Line crossing trackers."""

import colorsys
import math
import random
from collections import defaultdict, deque
from typing import List, Optional, Sequence, Tuple

from savant_rs.primitives.geometry import (
    IntersectionKind,
    Point,
    PolygonalArea,
    Segment,
)


class TwoLinesCrossingTracker:
    """Determines the direction based on the order in which two lines are crossed.
    This is more reliable method in the case of a line at the frame boundary due to
    the jitter of the detected bounding box.
    """

    def __init__(self, area: PolygonalArea):
        self._area = area
        self._prev_cross_edge_label = {}
        self._track_last_points = defaultdict(lambda: deque(maxlen=2))

    def remove_track(self, track_id: int):
        if track_id in self._track_last_points:
            del self._track_last_points[track_id]

    def add_track_point(self, track_id: int, point: Point):
        self._track_last_points[track_id].append(point)

    def check_tracks(self, track_ids: Sequence[int]) -> List[Optional[str]]:
        ret = [None] * len(track_ids)
        check_track_idxs = []
        segments = []
        for i, track_id in enumerate(track_ids):
            track_points = self._track_last_points[track_id]
            if len(track_points) == 2:
                segments.append(Segment(*track_points))
                check_track_idxs.append(i)

        cross_results = self._area.crossed_by_segments(segments)

        for cross_result, track_idx in zip(cross_results, check_track_idxs):
            if cross_result.kind in (IntersectionKind.Inside, IntersectionKind.Outside):
                continue

            track_id = track_ids[track_idx]
            cross_edge_labels = [edge[1] for edge in cross_result.edges]

            unlabeled_edge_crossed = False
            for edge_label in cross_edge_labels:
                if not edge_label:
                    unlabeled_edge_crossed = True
                    break

            if unlabeled_edge_crossed:
                continue

            if cross_result.kind == IntersectionKind.Enter:
                self._prev_cross_edge_label[track_id] = cross_edge_labels
                ret[track_idx] = '+'.join(['entry'] + cross_edge_labels)
                continue

            if cross_result.kind == IntersectionKind.Leave:
                if track_id in self._prev_cross_edge_label:
                    cross_edge_labels = (
                        self._prev_cross_edge_label[track_id] + cross_edge_labels
                    )
                else:
                    ret[track_idx] = '+'.join(['exit'] + cross_edge_labels)

            if len(set(cross_edge_labels)) == 2:
                # the track exited the area through a different edge than it entered
                ret[track_idx] = '->'.join(cross_edge_labels)

        return ret


class RandColorIterator:
    def __init__(self) -> None:
        self.golden_ratio_conjugate = 0.618033988749895
        self.hue = random.random()
        self.saturation = 0.7
        self.value = 0.95

    def __next__(self) -> Tuple[int, int, int, int]:
        self.hue = math.fmod(self.hue + 0.618033988749895, 1)
        rgb = colorsys.hsv_to_rgb(self.hue, self.saturation, self.value)
        r, g, b = map(lambda x: int(255 * x), rgb)
        return (r, g, b) + (255,)
