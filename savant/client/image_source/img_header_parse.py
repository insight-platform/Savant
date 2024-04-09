"""Image header parse utility."""

import re
from os import PathLike
from typing import BinaryIO, Tuple, Union

import magic

# positive lookbehind is included to avoid matching density in the JPEG header
PATTERN = re.compile(r'(?<=, )(?P<width>\d+)( x |x)(?P<height>\d+)')


def get_image_size_codec(file: Union[str, PathLike, BinaryIO]) -> Tuple[int, int, str]:
    """Get JPEG or PNG image width and height by parsing the file header.
    :param file: Path to an image file or a file handle to an image file opened as binary.
    :return: Image width, height and codec.
    """
    if hasattr(file, 'read') and hasattr(file, 'seek'):
        magic_out = magic.from_buffer(file.read(2048))
        file.seek(0)
    elif isinstance(file, (str, PathLike)):
        magic_out = magic.from_file(file)
    else:
        raise ValueError('File path or file handle is expected.')

    # codec str should correspond to savant.gstreamer.codecs.Codec enum values
    # can't import directly because of extra dependencies (gstreamer)
    # that aren't going to be present in all the adapter images
    if magic_out.startswith('JPEG image data'):
        codec = 'jpeg'
    elif magic_out.startswith('PNG image data'):
        codec = 'png'
    else:
        raise ValueError('Not a JPEG or PNG file.')

    match = PATTERN.search(magic_out)
    if match:
        return int(match['width']), int(match['height']), codec
    raise ValueError('Failed to get image size from image header.')
