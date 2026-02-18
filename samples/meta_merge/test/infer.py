"""Infer blackbox: receives frames, creates ROIs, runs YOLO, attaches detections to ROIs, forwards to sink."""

from __future__ import annotations

import os
import sys

from savant_rs.logging import LogLevel, log, set_log_level
from savant_rs.py.api.enums import ExternalFrameType
from savant_rs.primitives import VideoFrameContent
from savant_rs.primitives.geometry import RBBox
from savant_rs.utils.serialization import Message
from savant_rs.zmq import (
    BlockingReader,
    BlockingWriter,
    ReaderConfigBuilder,
    ReaderResultMessage,
    ReaderResultTimeout,
    WriterConfigBuilder,
    WriterResultSuccess,
)

set_log_level(LogLevel.Info)

ROI_NAMESPACE = 'router'
LEFT_ROI_LABEL = 'left_roi'
RIGHT_ROI_LABEL = 'right_roi'
PERSON_LABEL = 'person'
PERSON_NAMESPACE = 'infer'


YOLO_MODEL = os.environ.get('YOLO_MODEL', '/opt/models/yolo11m.pt')


def run_yolo_person_detection(
    jpeg_bytes: bytes,
) -> list[tuple[float, float, float, float, float]]:
    """Run YOLOv11 on JPEG, return list of (l, t, r, b, conf) for person class."""
    import io

    from PIL import Image
    from ultralytics import YOLO

    model = YOLO(YOLO_MODEL)
    img = Image.open(io.BytesIO(jpeg_bytes))
    results = model(img, verbose=False, device=0)

    persons: list[tuple[float, float, float, float, float]] = []
    for r in results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            cls_id = int(box.cls[0])
            if cls_id != 0:  # COCO person class
                continue
            xyxy = box.xyxy[0]
            l, t, r, b = float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])
            conf = float(box.conf[0])
            persons.append((l, t, r, b, conf))
    return persons


def process_frame(frame) -> None:
    """Add ROIs and person detections to frame."""
    from savant_rs.match_query import MatchQuery

    width = frame.width
    height = frame.height

    frame.delete_objects(MatchQuery.not_(MatchQuery.parent_defined()))

    left_bbox = RBBox.ltwh(0, 0, width / 2, height)
    left_roi = frame.create_object(
        namespace=ROI_NAMESPACE,
        label=LEFT_ROI_LABEL,
        detection_box=left_bbox,
    )

    right_bbox = RBBox.ltwh(width / 2, 0, width / 2, height)
    right_roi = frame.create_object(
        namespace=ROI_NAMESPACE,
        label=RIGHT_ROI_LABEL,
        detection_box=right_bbox,
    )

    jpeg_bytes = frame.content.get_data()
    persons = run_yolo_person_detection(jpeg_bytes)

    mid_x = width / 2
    for l, t, r, b, conf in persons:
        center_x = (l + r) / 2
        bbox = RBBox.ltwh(l, t, r - l, b - t)
        if center_x < mid_x:
            parent = left_roi
        else:
            parent = right_roi
        frame.create_object(
            namespace=PERSON_NAMESPACE,
            label=PERSON_LABEL,
            detection_box=bbox,
            parent_id=parent.id,
            confidence=conf,
        )


def main() -> int:
    ingress_socket = os.environ.get(
        'ZMQ_INGRESS', 'router+bind:ipc:///tmp/zmq-sockets/infer.ipc'
    )
    egress_socket = os.environ.get(
        'ZMQ_EGRESS', 'dealer+bind:ipc:///tmp/zmq-sockets/sink.ipc'
    )

    receive_timeout_ms = int(os.environ.get('ZMQ_RECEIVE_TIMEOUT_MS', '5000'))
    send_timeout_ms = int(os.environ.get('ZMQ_SEND_TIMEOUT_MS', '10000'))
    send_retries = int(os.environ.get('ZMQ_SEND_RETRIES', '10'))
    receive_retries = int(os.environ.get('ZMQ_RECEIVE_RETRIES', '10'))

    reader_builder = ReaderConfigBuilder(ingress_socket)
    reader_builder.with_receive_timeout(receive_timeout_ms)
    reader_config = reader_builder.build()
    reader = BlockingReader(reader_config)
    reader.start()

    writer_builder = WriterConfigBuilder(egress_socket)
    writer_builder.with_send_timeout(send_timeout_ms)
    writer_builder.with_receive_timeout(receive_timeout_ms)
    writer_builder.with_send_retries(send_retries)
    writer_builder.with_receive_retries(receive_retries)
    writer_config = writer_builder.build()
    writer = BlockingWriter(writer_config)
    writer.start()

    try:
        while True:
            res = reader.receive()

            if isinstance(res, ReaderResultTimeout):
                continue

            if isinstance(res, ReaderResultMessage):
                msg = res.message
                if msg.is_end_of_stream():
                    eos = msg.as_end_of_stream()
                    if eos is not None:
                        writer.send_eos(eos.source_id)
                    break
                if msg.is_video_frame():
                    frame = msg.as_video_frame()
                    if frame is not None:
                        jpeg_bytes = res.data(0) if res.data_len() > 0 else b''
                        if not jpeg_bytes:
                            log(LogLevel.Error, 'infer', 'Video frame has no image data in extra')
                            return 1
                        frame.content = VideoFrameContent.internal(jpeg_bytes)
                        process_frame(frame)
                        frame.content = VideoFrameContent.external(
                            ExternalFrameType.ZEROMQ.value, None
                        )
                        out_msg = Message.video_frame(frame)
                        send_res = writer.send_message(frame.source_id, out_msg, jpeg_bytes)
                        if not isinstance(send_res, WriterResultSuccess):
                            log(LogLevel.Error, 'infer', f'Failed to send frame: {send_res}')
                            return 1
    finally:
        reader.shutdown()
        writer.shutdown()

    return 0


if __name__ == '__main__':
    sys.exit(main())
