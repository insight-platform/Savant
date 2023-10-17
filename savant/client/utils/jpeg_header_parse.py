"""JPEG header parse utility."""
import re
from os import PathLike
from typing import BinaryIO, Tuple, Union

import magic

PATTERN = re.compile(r'(?<=, )(?P<width>\d+)x(?P<height>\d+)')


def get_jpeg_size(file: Union[str, PathLike, BinaryIO]) -> Tuple[int, int]:
    """Get JPEG image width and height by parsing the file header.
    :param file: Path to a JPEG file or a file handle to a JPEG file opened as binary.
    :return: Image width and height.
    """
    if hasattr(file, 'read') and hasattr(file, 'seek'):
        magic_out = magic.from_buffer(file.read(2048))
        file.seek(0)
    elif isinstance(file, (str,  PathLike)):
        magic_out = magic.from_file(file)
    else:
        raise ValueError('File path or file handle is expected.')
    if magic_out.startswith('JPEG image data'):
        out = PATTERN.search(magic_out)
        if out:
            return int(out['width']), int(out['height'])
        raise ValueError('Failed to get image size from JPEG header.')
    raise ValueError('Not a JPEG file')
