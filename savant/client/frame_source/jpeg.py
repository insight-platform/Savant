import os
from os import PathLike
from typing import BinaryIO, List, Optional, Tuple, Union

from savant_rs.primitives import (
    Attribute,
    AttributeValue,
    VideoFrame,
    VideoFrameContent,
    VideoFrameUpdate,
)

from savant.api.constants import DEFAULT_NAMESPACE
from savant.api.enums import ExternalFrameType
from savant.client.frame_source import FrameSource
from savant.client.utils import get_jpeg_size
from savant.utils.logging import get_logger

SECOND_IN_NS = 10**9

logger = get_logger(__name__)


class JpegSource(FrameSource):
    """Frame source for JPEG files.

    :param source_id: Source ID.
    :param filepath: Path to a JPEG file.
    :param file_handle: File handle to a JPEG file.
    :param pts: Frame presentation timestamp.
    :param framerate: Framerate (numerator, denominator).
    """

    def __init__(
        self,
        source_id: str,
        filepath: Union[str, PathLike],
        file_handle: Optional[BinaryIO] = None,
        pts: int = 0,
        framerate: Tuple[int, int] = (30, 1),
        updates: Optional[List[VideoFrameUpdate]] = None,
    ):
        if not file_handle and not os.path.exists(filepath):
            raise ValueError(f'File {filepath!r} does not exist.')

        self._source_id = source_id
        self._filepath = filepath
        self._file_handle = file_handle
        self._pts = pts
        self._framerate = framerate
        self._updates = updates or []
        self._duration = SECOND_IN_NS * framerate[1] // framerate[0]
        self._time_base = (1, SECOND_IN_NS)

    @property
    def source_id(self) -> str:
        """Source ID."""
        return self._source_id

    @property
    def filepath(self) -> Union[str, PathLike]:
        """Path to a JPEG file."""
        return self._filepath

    @property
    def pts(self) -> int:
        """Frame presentation timestamp."""
        return self._pts

    @property
    def framerate(self) -> Tuple[int, int]:
        """Framerate."""
        return self._framerate

    @property
    def updates(self) -> List[VideoFrameUpdate]:
        """List of frame updates."""
        return self._updates

    @property
    def duration(self) -> int:
        """Frame duration."""
        return self._duration

    def with_pts(self, pts: int) -> 'FrameSource':
        """Set frame presentation timestamp."""
        return self._update_param('pts', pts)

    def with_framerate(self, framerate: Tuple[int, int]) -> 'FrameSource':
        """Set framerate."""
        return self._update_param('framerate', framerate)

    def with_update(self, update: VideoFrameUpdate) -> 'JpegSource':
        return self._update_param('updates', self._updates + [update])

    def build_frame(self) -> Tuple[VideoFrame, bytes]:
        width, height = get_jpeg_size(self._filepath)

        if self._file_handle:
            content = self._file_handle.read()
        else:
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
        logger.debug(
            'Built video frame %s/%s from file %s.',
            video_frame.source_id,
            video_frame.pts,
            self._filepath,
        )

        return video_frame, content

    def _update_param(self, name, value) -> 'JpegSource':
        return JpegSource(
            **{
                'source_id': self._source_id,
                'filepath': self._filepath,
                'file_handle': self._file_handle,
                'pts': self._pts,
                'framerate': self._framerate,
                'updates': self._updates,
                name: value,
            }
        )

    def __repr__(self):
        return (
            f'JpegSource('
            f'source_id={self._source_id}, '
            f'filepath={self._filepath}, '
            f'pts={self._pts})'
        )
