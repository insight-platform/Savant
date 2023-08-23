import os
from os import PathLike
from typing import List, Tuple, Union

import cv2
from savant_rs.primitives import (
    Attribute,
    AttributeValue,
    VideoFrame,
    VideoFrameContent,
    VideoFrameUpdate,
)

from savant.api.constants import DEFAULT_NAMESPACE
from savant.api.enums import ExternalFrameType
from savant.source_sink_framework.frame_source import FrameSource


class JpegSource(FrameSource):
    def __init__(
        self,
        source_id: str,
        filepath: Union[str, PathLike],
        updates: List[VideoFrameUpdate] = None,
    ):
        if not os.path.exists(filepath):
            raise ValueError(f'File {filepath!r} does not exist')
        self._source_id = source_id
        self._filepath = filepath
        self._updates = updates or []
        self._framerate = '30/1'
        self._duration = 10**9 // 30
        self._time_base = (1, 10**9)

    def with_update(self, update: VideoFrameUpdate) -> 'JpegSource':
        return JpegSource(
            self._source_id,
            self._filepath,
            self._updates + [update],
        )

    def build_frame(self) -> Tuple[VideoFrame, bytes]:
        # TODO: get image size without decoding
        img = cv2.imread(self._filepath)
        height, width, _ = img.shape
        with open(self._filepath, 'rb') as f:
            content = f.read()
        video_frame = VideoFrame(
            source_id=self._source_id,
            framerate=self._framerate,
            codec='jpeg',
            width=width,
            height=height,
            content=VideoFrameContent.external(ExternalFrameType.ZEROMQ.value, None),
            keyframe=True,
            pts=0,
            duration=self._duration,
            time_base=self._time_base,
        )
        video_frame.set_attribute(
            Attribute(
                namespace=DEFAULT_NAMESPACE,
                name='location',
                values=[AttributeValue.string(str(self._filepath))],
            )
        )
        for update in self._updates:
            video_frame.update(update)

        return video_frame, content
