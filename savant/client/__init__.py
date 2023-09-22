"""Source/Sink framework for development and QA purposes."""

from .builder.sink import SinkBuilder
from .builder.source import SourceBuilder
from .frame_source import FrameSource
from .log_provider import LogProvider
from .log_provider.jaeger import JaegerLogProvider

__all__ = [
    'SinkBuilder',
    'SourceBuilder',
    'FrameSource',
    'LogProvider',
    'JaegerLogProvider',
]

try:
    from .frame_source.jpeg import JpegSource

    __all__.append('JpegSource')
except ImportError:
    from savant.utils.logging import get_logger

    logger = get_logger(__name__)
    logger.warning('Failed to import JpegSource. Install OpenCV to enable it.')
