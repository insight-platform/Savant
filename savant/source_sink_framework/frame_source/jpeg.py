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

SECOND_IN_NS = 10**9


class JpegSource(FrameSource):
    def __init__(
        self,
        source_id: str,
        filepath: Union[str, PathLike],
        pts: int = 0,
        framerate: Tuple[int, int] = (30, 1),
        updates: List[VideoFrameUpdate] = None,
    ):
        if not os.path.exists(filepath):
            raise ValueError(f'File {filepath!r} does not exist')
        self._source_id = source_id
        self._filepath = filepath
        self._pts = pts
        self._framerate = framerate
        self._updates = updates or []
        self._duration = SECOND_IN_NS * framerate[1] // framerate[0]
        self._time_base = (1, SECOND_IN_NS)

    @property
    def source_id(self) -> str:
        return self._source_id

    @property
    def filepath(self) -> Union[str, PathLike]:
        return self._filepath

    @property
    def pts(self) -> int:
        return self._pts

    @property
    def framerate(self) -> Tuple[int, int]:
        return self._framerate

    @property
    def updates(self) -> List[VideoFrameUpdate]:
        return self._updates

    @property
    def duration(self) -> int:
        return self._duration

    def with_pts(self, pts: int) -> 'FrameSource':
        return self._update_param('pts', pts)

    def with_framerate(self, framerate: Tuple[int, int]) -> 'FrameSource':
        return self._update_param('framerate', framerate)

    def with_update(self, update: VideoFrameUpdate) -> 'JpegSource':
        return self._update_param('updates', self._updates + [update])

    def build_frame(self) -> Tuple[VideoFrame, bytes]:
        # TODO: get image size without decoding
        img = cv2.imread(self._filepath)
        height, width, _ = img.shape
        with open(self._filepath, 'rb') as f:
            content = f.read()
        video_frame = VideoFrame(
            source_id=self._source_id,
            framerate=f'{self._framerate[0]}/{self._framerate[1]}',
            codec='jpeg',
            width=width,
            height=height,
            content=VideoFrameContent.external(ExternalFrameType.ZEROMQ.value, None),
            keyframe=True,
            pts=self._pts,
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

    def _update_param(self, name, value) -> 'JpegSource':
        return JpegSource(
            **{
                'source_id': self._source_id,
                'filepath': self._filepath,
                'pts': self._pts,
                'framerate': self._framerate,
                'updates': self._updates,
                name: value,
            }
        )
