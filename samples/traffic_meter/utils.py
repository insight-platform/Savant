"""Line crossing trackers."""
from collections import deque, defaultdict, namedtuple
from enum import Enum
from typing import Optional, Tuple, List
import random
import math
import numpy as np
from savant_rs.primitives import PolygonalArea, Segment, IntersectionKind
from savant_rs.primitives import Point as SavantRsPoint


class Direction(Enum):
    entry = 0
    exit = 1


Point = namedtuple('Point', ['x', 'y'])


def get_segment_intersection_point(
    a1: Point, a2: Point, b1: Point, b2: Point
) -> Optional[Point]:
    """Finds the intersection of line segments (a1, a2) and (b1, b2).
    Segments intersect if there is a point of intersection of the lines of the segments
    and the point belongs to the segments.
    line by (x1, y1) and (x2, y2): (x - x1) / (x2 - x1) = (y - y1) / (y2 - y1)
    """
    ax1, ay1 = a1
    ax2, ay2 = a2
    bx1, by1 = b1
    bx2, by2 = b2
    denominator = (ax1 - ax2) * (by1 - by2) - (ay1 - ay2) * (bx1 - bx2)
    if denominator == 0:
        return None
    x = (
        (ax1 * ay2 - ay1 * ax2) * (bx1 - bx2) - (ax1 - ax2) * (bx1 * by2 - by1 * bx2)
    ) / denominator
    y = (
        (ax1 * ay2 - ay1 * ax2) * (by1 - by2) - (ay1 - ay2) * (bx1 * by2 - by1 * bx2)
    ) / denominator
    if (
        (x - ax1) * (x - ax2) <= 0
        and (y - ay1) * (y - ay2) <= 0
        and (x - bx1) * (x - bx2) <= 0
        and (y - by1) * (y - by2) <= 0
    ):
        return Point(x, y)
    return None


def get_clockwise_direction(p0: Point, p1: Point, p2: Point) -> str:
    """https://algorithmtutor.com/Computational-Geometry/Determining-if-two-consecutive-segments-turn-left-or-right/"""
    p0p1 = Point(p1.x - p0.x, p1.y - p0.y)
    p0p2 = Point(p2.x - p0.x, p2.y - p0.y)
    return 'right' if p0p1.x * p0p2.y - p0p2.x * p0p1.y > 0 else 'left'


class LineCrossingTracker:
    """Checks the intersection of two segments, given line and segment,
    built on the last two points of the track.
    Determines if the direction of the track at the intersection point
    (calculated clockwise) matches the specified entry direction.
    """

    def __init__(self, line: Tuple[Point, Point], entry_direction: str = 'left'):
        self._lines = [line]
        self._entry_direction = entry_direction
        # {track_id: (p1, p2)}
        self._track_last_points = defaultdict(lambda: deque(maxlen=2))

    @property
    def lines(self):
        return self._lines

    def remove_track(self, track_id: int):
        if track_id in self._track_last_points:
            del self._track_last_points[track_id]

    def add_track_point(self, track_id: int, point: Point):
        self._track_last_points[track_id].append(point)

    def check_track(self, track_id: int) -> Optional[Direction]:
        if len(self._track_last_points[track_id]) > 1:
            intersection_point = get_segment_intersection_point(
                *self._lines[0], *self._track_last_points[track_id]
            )
            if intersection_point:
                clockwise_direction = get_clockwise_direction(
                    intersection_point,
                    self._track_last_points[track_id][1],
                    self._lines[0][1],
                )
                return (
                    Direction.entry
                    if clockwise_direction == self._entry_direction
                    else Direction.exit
                )
        return None


def point_line_distance(point: Point, line: Tuple[Point, Point]):
    """Calculate distance between a point and a line defined by two points."""
    pt_a = np.asarray(line[0])
    pt_b = np.asarray(line[1])
    pt_c = np.asarray(point)
    return np.abs(np.cross(pt_b - pt_a, pt_a - pt_c)) / np.linalg.norm(pt_b - pt_a)


class TwoLinesCrossingTracker(LineCrossingTracker):
    """Determines the direction based on the order in which two lines are crossed.
    This is more reliable method in the case of a line at the frame boundary due to
    the jitter of the detected bounding box.
    """

    def __init__(self, line1: Tuple[Point, Point], line2: Tuple[Point, Point]):
        super().__init__(line1)
        self._lines.append(line2)
        # {track_id: (c1, c2)} - stores the last 2 crossing lines
        self._line_crossings = defaultdict(lambda: deque(maxlen=2))

        l1_pt_1, l2_pt_2 = line1
        l1_pt_2, l2_pt_1 = line2
        rs_pt1 = SavantRsPoint(*l1_pt_1)
        rs_pt2 = SavantRsPoint(*l1_pt_2)
        rs_pt3 = SavantRsPoint(*l2_pt_1)
        rs_pt4 = SavantRsPoint(*l2_pt_2)
        self.area = PolygonalArea(
            [rs_pt1, rs_pt2, rs_pt3, rs_pt4], [None, "entry", None, "exit"]
        )
        self.rs_line_crossings = {}

    def check_track_rs(self, track_id: int) -> Optional[Direction]:
        track_points = self._track_last_points[track_id]
        if len(track_points) != 2:
            return None

        segment = Segment(
            SavantRsPoint(*track_points[0]), SavantRsPoint(*track_points[1])
        )
        rs_result = self.area.crossed_by_segment(segment)

        if rs_result.kind == IntersectionKind.Enter:
            self.rs_line_crossings[track_id] = rs_result.edges[0]

        elif rs_result.kind == IntersectionKind.Leave:
            if track_id in self.rs_line_crossings:
                if self.rs_line_crossings[track_id] == rs_result.edges[0]:
                    return None
                if rs_result.edges == [(1, 'entry')]:
                    return Direction.entry
                if rs_result.edges == [(3, 'exit')]:
                    return Direction.exit
                return None
            raise ValueError(f"Unexpected leave no prev enter: {rs_result.edges}")

        elif rs_result.kind == IntersectionKind.Cross:
            if rs_result.edges == [(3, 'exit'), (1, 'entry')]:
                return Direction.entry
            if rs_result.edges == [(1, 'entry'), (3, 'exit')]:
                return Direction.exit
            return None

        else:
            # rs_result.kind == IntersectionKind.Inside
            # rs_result.kind == IntersectionKind.Outside
            return None
        
    def check_track_rs_batch(self, track_ids: List[int]) -> List[Optional[Direction]]:
        ret = [None] * len(track_ids)
        check_track_idxs = []
        segments = []
        for i, track_id in enumerate(track_ids):
            track_points = self._track_last_points[track_id]
            if len(track_points) == 2:
                segments.append(
                    Segment(
                        SavantRsPoint(*track_points[0]), SavantRsPoint(*track_points[1])
                    )
                )
                check_track_idxs.append(i)

        cross_results = self.area.crossed_by_segments(segments)

        for cross_result, track_idx in zip(cross_results, check_track_idxs):

            track_id = track_ids[track_idx]

            if cross_result.kind == IntersectionKind.Enter:
                self.rs_line_crossings[track_id] = cross_result.edges[0]

            elif cross_result.kind == IntersectionKind.Leave:
                # if the defined area was entered before
                # and the leave edge is not the same as the enter edge
                if track_id in self.rs_line_crossings and self.rs_line_crossings[track_id] != cross_result.edges[0]:
                    if cross_result.edges == [(1, 'entry')]:
                        ret[track_idx] = Direction.entry
                    elif cross_result.edges == [(3, 'exit')]:
                        ret[track_idx] = Direction.exit

            elif cross_result.kind == IntersectionKind.Cross:
                if cross_result.edges == [(3, 'exit'), (1, 'entry')]:
                    ret[track_idx] = Direction.entry
                elif cross_result.edges == [(1, 'entry'), (3, 'exit')]:
                    ret[track_idx] = Direction.exit

        return ret

    def check_track(self, track_id: int) -> Optional[Direction]:
        track_points = self._track_last_points[track_id]
        if len(track_points) != 2:
            return None

        step_intersections = 0
        for line_idx, line in enumerate(self._lines):
            inter_pt = get_segment_intersection_point(*line, *track_points)
            if inter_pt:
                self._line_crossings[track_id].append(line_idx)
                step_intersections += 1

                track_line_crossings = self._line_crossings[track_id]
                if len(track_line_crossings) == 2:
                    first_cross_line_idx = track_line_crossings[0]
                    second_cross_line_idx = track_line_crossings[1]
                    if first_cross_line_idx == second_cross_line_idx:
                        return None

                    if step_intersections == 2:
                        prev_position = track_points[0]
                        dist_first_cross = point_line_distance(
                            prev_position, self._lines[first_cross_line_idx]
                        )
                        dist_second_cross = point_line_distance(
                            prev_position, self._lines[second_cross_line_idx]
                        )
                        if dist_first_cross > dist_second_cross:
                            first_cross_line_idx, second_cross_line_idx = (
                                second_cross_line_idx,
                                first_cross_line_idx,
                            )

                    return (
                        Direction.entry
                        if first_cross_line_idx < second_cross_line_idx
                        else Direction.exit
                    )

        return None


class RandColorIterator:
    def __init__(self) -> None:
        self.golden_ratio_conjugate = 0.618033988749895
        self.hue = random.random()
        self.saturation = 0.7
        self.value = 0.95

    def __next__(self):
        self.hue = math.fmod(self.hue + 0.618033988749895, 1)
        return hsv_to_rgb(self.hue, self.saturation, self.value)


def hsv_to_rgb(h, s, v):
    """HSV values in [0..1]
    returns [r, g, b] values in [0..1]
    """
    h_i = int(h * 6)
    f = h * 6 - h_i
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    if h_i == 0:
        r, g, b = v, t, p
    elif h_i == 1:
        r, g, b = q, v, p
    elif h_i == 2:
        r, g, b = p, v, t
    elif h_i == 3:
        r, g, b = p, q, v
    elif h_i == 4:
        r, g, b = t, p, v
    elif h_i == 5:
        r, g, b = v, p, q
    return r, g, b
