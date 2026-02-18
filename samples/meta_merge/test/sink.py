"""Sink: receives frames, compares detected objects with (source, person) attribute using IoU."""

from __future__ import annotations

import os
import sys

from savant_rs.logging import LogLevel, log, set_log_level
from savant_rs.match_query import MatchQuery, StringExpression
from savant_rs.primitives.geometry import RBBox
from savant_rs.zmq import (
    BlockingReader,
    ReaderConfigBuilder,
    ReaderResultMessage,
    ReaderResultTimeout,
)

set_log_level(LogLevel.Info)

SOURCE_NAMESPACE = 'source'
PERSON_ATTR = 'person'
IOU_THRESHOLD = 0.5


def parse_expected_persons(frame) -> list[tuple[RBBox, float]]:
    """Extract (source, person) attribute: list of (bbox, conf) from AttributeValue list."""
    attr = frame.get_attribute(SOURCE_NAMESPACE, PERSON_ATTR)
    if attr is None:
        return []

    result: list[tuple[RBBox, float]] = []
    values = attr.values_view
    i = 0
    while i + 1 < len(values):
        av_box = values[i]
        av_conf = values[i + 1]
        box_ints = av_box.as_integers()
        conf = av_conf.as_float()
        if box_ints is not None and len(box_ints) >= 4 and conf is not None:
            l, t, r, b = box_ints[0], box_ints[1], box_ints[2], box_ints[3]
            bbox = RBBox.ltrb(float(l), float(t), float(r), float(b))
            result.append((bbox, conf))
        i += 2
    return result


def get_detected_persons(frame) -> list[RBBox]:
    """Extract person detection boxes from ROI children (infer.person)."""
    detected: list[RBBox] = []
    for obj in frame.access_objects(MatchQuery.label(StringExpression.eq('person'))):
        detected.append(obj.detection_box)
    return detected


def compare_with_iou(
    frame_uuid: str,
    expected: list[tuple[RBBox, float]],
    detected: list[RBBox],
) -> bool:
    """Compare expected vs detected using IoU. Returns True if match, False on error."""
    used = [False] * len(detected)
    for exp_bbox, _ in expected:
        best_iou = 0.0
        best_idx = -1
        for i, det_bbox in enumerate(detected):
            if used[i]:
                continue
            iou = exp_bbox.iou(det_bbox)
            if iou > best_iou:
                best_iou = iou
                best_idx = i
        if best_iou < IOU_THRESHOLD or best_idx < 0:
            log(
                LogLevel.Error,
                'sink',
                f'Expected box {exp_bbox} has no matching detection (best IoU={best_iou:.3f})',
            )
            return False
        log(
            LogLevel.Info,
            'sink',
            f'match: frame_uuid={frame_uuid} attribute_object={exp_bbox} inferred_object={detected[best_idx]} score={best_iou:.4f}',
        )
        used[best_idx] = True

    for i, u in enumerate(used):
        if not u:
            log(
                LogLevel.Error,
                'sink',
                f'Extra detection {detected[i]} has no expected match',
            )
            return False
    return True


def main() -> int:
    socket = os.environ.get(
        'ZMQ_SOCKET', 'router+connect:ipc:///tmp/zmq-sockets/sink.ipc'
    )
    receive_timeout_ms = int(os.environ.get('ZMQ_RECEIVE_TIMEOUT_MS', '5000'))

    reader_builder = ReaderConfigBuilder(socket)
    reader_builder.with_receive_timeout(receive_timeout_ms)
    reader_config = reader_builder.build()
    reader = BlockingReader(reader_config)
    reader.start()

    errors = 0
    try:
        while True:
            res = reader.receive()

            if isinstance(res, ReaderResultTimeout):
                continue

            if isinstance(res, ReaderResultMessage):
                msg = res.message
                if msg.is_end_of_stream():
                    break
                if msg.is_video_frame():
                    frame = msg.as_video_frame()
                    if frame is not None:
                        expected = parse_expected_persons(frame)
                        detected = get_detected_persons(frame)
                        if not compare_with_iou(frame.uuid, expected, detected):
                            errors += 1
    finally:
        reader.shutdown()

    if errors > 0:
        log(LogLevel.Error, 'sink', f'FAILED: {errors} frame(s) had detection mismatches')
        return 1
    log(LogLevel.Info, 'sink', 'PASSED: All detections matched expected (source, person) attribute')
    return 0


if __name__ == '__main__':
    sys.exit(main())
