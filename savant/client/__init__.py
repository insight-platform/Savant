"""Source/Sink framework for development and QA purposes."""
from .builder.sink import SinkBuilder
from .builder.source import SourceBuilder
from .frame_source import FrameSource
from .log_provider import LogProvider
from .log_provider.jaeger import JaegerLogProvider

__all__ = [
    'SinkBuilder',
    'SourceBuilder',
    'LogProvider',
    'FrameSource',
    'JaegerLogProvider',
]

try:
    from .image_source import JpegSource, PngSource

    __all__ += ['JpegSource', 'PngSource']
except ModuleNotFoundError:
    from savant.utils.logging import get_logger

    logger = get_logger(__name__)
    logger.warning(
        'Image sources "python-magic" dependency is missing. '
        'JpegSource and PngSource are not available.'
    )
