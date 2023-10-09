import math
import os
from os import PathLike
from typing import List, Optional, Tuple, Union

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
from savant.client.frame_source import FrameSource
from savant.utils.logging import get_logger

SECOND_IN_NS = 10**9

logger = get_logger(__name__)


class JpegSource(FrameSource):
    """Frame source for JPEG files.

    :param source_id: Source ID.
    :param filepath: Path to a JPEG file.
    :param pts: Frame presentation timestamp.
    :param framerate: Framerate (numerator, denominator).
    :param source_id_add_size_suffix: Whether to add a suffix
        of a form width-height to the source ID.
    :param img_size: Target image size (width, height),
        None if the original image size should be used.
    """

    def __init__(
        self,
        source_id: str,
        filepath: Union[str, PathLike],
        pts: int = 0,
        framerate: Tuple[int, int] = (30, 1),
        source_id_add_size_suffix: bool = False,
        img_size: Optional[Tuple[int, int]] = None,
        updates: Optional[List[VideoFrameUpdate]] = None,
    ):
        if not os.path.exists(filepath):
            raise ValueError(f'File {filepath!r} does not exist')
        self._filepath = filepath

        if img_size is not None:
            self._width, self._height = img_size
        else:
            # TODO: get image size without decoding
            img = cv2.imread(self._filepath)
            self._height, self._width, _ = img.shape

        self._source_id = source_id
        self._source_id_add_size_suffix = source_id_add_size_suffix
        self._pts = pts
        self._framerate = framerate
        self._updates = updates or []
        self._duration = SECOND_IN_NS * framerate[1] // framerate[0]
        self._time_base = (1, SECOND_IN_NS)

    @property
    def source_id(self) -> str:
        """Source ID."""
        if self._source_id_add_size_suffix:
            return f'{self._source_id}-{self._width}-{self._height}'
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

    @property
    def img_size(self) -> Tuple[int, int]:
        """Target image size (width, height)."""
        return self._width, self._height

    def with_pts(self, pts: int) -> 'FrameSource':
        """Set frame presentation timestamp."""
        return self._update_param('pts', pts)

    def with_framerate(self, framerate: Tuple[int, int]) -> 'FrameSource':
        """Set framerate."""
        return self._update_param('framerate', framerate)

    def with_update(self, update: VideoFrameUpdate) -> 'JpegSource':
        return self._update_param('updates', self._updates + [update])

    def with_aspect_ratio(self, aspect: Tuple[int, int]) -> 'JpegSource':
        new_aspect = aspect[0] / aspect[1]
        current_aspect = self._width / self._height

        if math.isclose(current_aspect, new_aspect, rel_tol=1e-3):
            new_width = self._width
            new_height = self._height
        elif current_aspect < new_aspect:
            # increase width
            new_width = round(self._height * aspect[0] / aspect[1])
            new_height = self._height
        else:
            # increase height
            new_width = self._width
            new_height = round(self._width * aspect[1] / aspect[0])

        return self._update_param('img_size', (new_width, new_height))

    def with_source_id_add_size_suffix(self, val: bool) -> 'JpegSource':
        return self._update_param('source_id_add_size_suffix', val)

    def build_frame(self) -> Tuple[VideoFrame, bytes]:
        # TODO: get image size without decoding
        img = cv2.imread(self._filepath)
        height, width, channels = img.shape

        if self._height >= height and self._width >= width:
            # add padding
            if self._height > height:
                top = (self._height - height) // 2
                bottom = self._height - height - top
            else:
                top = bottom = 0
            if self._width > width:
                left = (self._width - width) // 2
                right = self._width - width - left
            else:
                left = right = 0

            img = cv2.copyMakeBorder(
                img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0] * channels
            )
        else:
            # resize
            img = cv2.resize(img, (self._width, self._height))

        _, buf = cv2.imencode('.jpeg', img)
        content = buf.tobytes()

        video_frame = VideoFrame(
            source_id=self.source_id,
            framerate=f'{self._framerate[0]}/{self._framerate[1]}',
            codec='jpeg',
            width=self._width,
            height=self._height,
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
                'pts': self._pts,
                'framerate': self._framerate,
                'updates': self._updates,
                'img_size': (self._width, self._height),
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
