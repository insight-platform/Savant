"""Source service: loads JPEG, runs YOLO, stores person boxes, sends frames via BlockingWriter."""

from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path

from savant_rs.logging import LogLevel, log, set_log_level
from savant_rs.py.api.enums import ExternalFrameType
from savant_rs.primitives import (
    AttributeValue,
    VideoFrame,
    VideoFrameContent,
    VideoFrameTranscodingMethod,
)

# Image sent as extra argument to send_message, not in frame content
from savant_rs.utils.serialization import Message
from savant_rs.zmq import BlockingWriter, WriterConfigBuilder, WriterResultSuccess

set_log_level(LogLevel.Info)

SOURCE_NAMESPACE = 'source'
PERSON_ATTR = 'person'


def merge_jpeg_side_by_side(
    image_path: str,
    target_width: int | None = None,
    target_height: int | None = None,
) -> tuple[bytes, int, int]:
    """Merge JPEG side-by-side (L|R) and return merged bytes, width, height."""
    from PIL import Image

    with Image.open(image_path) as img:
        img = img.convert('RGB')
        w, h = img.size
        merged = Image.new('RGB', (w * 2, h))
        merged.paste(img, (0, 0))
        merged.paste(img, (w, 0))

        if target_width is not None and target_height is not None:
            merged = merged.resize(
                (target_width, target_height), Image.Resampling.LANCZOS
            )

        with tempfile.NamedTemporaryFile(suffix='.jpeg', delete=False) as f:
            merged.save(f.name, 'JPEG', quality=85)
            with open(f.name, 'rb') as rf:
                data = rf.read()
            os.unlink(f.name)

        return data, merged.width, merged.height


YOLO_MODEL = os.environ.get('YOLO_MODEL', '/opt/models/yolo11m.pt')


def run_yolo_person_detection(
    jpeg_bytes: bytes,
) -> list[tuple[int, int, int, int, float]]:
    """Run YOLOv11 on JPEG, return list of (l, t, r, b, conf) for person class."""
    import io

    from PIL import Image
    from ultralytics import YOLO

    import torch

    model = YOLO(YOLO_MODEL)
    img = Image.open(io.BytesIO(jpeg_bytes))
    device = 0 if torch.cuda.is_available() else 'cpu'
    results = model(img, verbose=False, device=device)

    persons: list[tuple[int, int, int, int, float]] = []
    for r in results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            cls_id = int(box.cls[0])
            if cls_id != 0:  # COCO person class
                continue
            xyxy = box.xyxy[0]
            l, t, right, b = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
            conf = float(box.conf[0])
            persons.append((l, t, right, b, conf))
    return persons


def main() -> int:
    socket = os.environ.get(
        'ZMQ_SOCKET', 'dealer+connect:ipc:///tmp/zmq-sockets/infer.ipc'
    )
    source_id = os.environ.get('SOURCE_ID', 'test_source')
    repetitions = int(os.environ.get('REPETITIONS', '1'))
    image_path = os.environ.get('IMAGE_PATH', '/app/test_image.jpeg')
    frame_interval_ms = int(os.environ.get('FRAME_INTERVAL_MS', '0'))
    target_w = int(env) if (env := os.environ.get('FRAME_WIDTH')) else None
    target_h = int(env) if (env := os.environ.get('FRAME_HEIGHT')) else None

    if not Path(image_path).exists():
        log(LogLevel.Error, 'source', f'Image not found: {image_path}')
        return 1

    merged_bytes, width, height = merge_jpeg_side_by_side(
        image_path, target_width=target_w, target_height=target_h
    )
    persons = run_yolo_person_detection(merged_bytes)

    values: list[AttributeValue] = []
    for l, t, r, b, conf in persons:
        values.append(AttributeValue.integers([l, t, r, b]))
        values.append(AttributeValue.float(conf))

    send_timeout_ms = int(os.environ.get('ZMQ_SEND_TIMEOUT_MS', '10000'))
    receive_timeout_ms = int(os.environ.get('ZMQ_RECEIVE_TIMEOUT_MS', '5000'))
    send_retries = int(os.environ.get('ZMQ_SEND_RETRIES', '10'))
    receive_retries = int(os.environ.get('ZMQ_RECEIVE_RETRIES', '10'))

    builder = WriterConfigBuilder(socket)
    builder.with_send_timeout(send_timeout_ms)
    builder.with_receive_timeout(receive_timeout_ms)
    builder.with_send_retries(send_retries)
    builder.with_receive_retries(receive_retries)
    writer_config = builder.build()
    writer = BlockingWriter(writer_config)
    writer.start()

    try:
        for rep in range(repetitions):
            frame = VideoFrame(
                source_id=source_id,
                framerate='30/1',
                width=width,
                height=height,
                content=VideoFrameContent.external(
                    ExternalFrameType.ZEROMQ.value, None
                ),
                transcoding_method=VideoFrameTranscodingMethod.Copy,
                codec='jpeg',
                keyframe=True,
                pts=rep,
            )
            frame.set_persistent_attribute(
                namespace=SOURCE_NAMESPACE,
                name=PERSON_ATTR,
                values=values,
            )

            msg = Message.video_frame(frame)
            res = writer.send_message(source_id, msg, merged_bytes)
            if not isinstance(res, WriterResultSuccess):
                log(LogLevel.Error, 'source', f'Failed to send frame: {res}')
                return 1

            if frame_interval_ms > 0 and rep < repetitions - 1:
                time.sleep(frame_interval_ms / 1000.0)

        writer.send_eos(source_id)
        log(LogLevel.Info, 'source', f'Sent {repetitions} frame(s) + EOS')

        post_eos_idle_s = int(os.environ.get('POST_EOS_IDLE_S', '0'))
        if post_eos_idle_s > 0:
            log(
                LogLevel.Info,
                'source',
                f'Waiting {post_eos_idle_s}s for pipeline to drain (will be killed by compose)',
            )
            time.sleep(post_eos_idle_s)
    finally:
        writer.shutdown()

    return 0


if __name__ == '__main__':
    sys.exit(main())
