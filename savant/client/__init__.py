"""Source/Sink framework for development and QA purposes."""

from .builder.sink import SinkBuilder
from .builder.source import SourceBuilder
from .frame_source import FrameSource
from .log_provider import LogProvider
from .log_provider.jaeger import JaegerLogProvider

try:
    from .jpeg_source import JpegSource
except ImportError:
    import logging

    logging.warning('Failed to import JpegSource. Install OpenCV to enable it.')
