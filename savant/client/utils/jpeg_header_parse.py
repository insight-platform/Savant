"""JPEG header parse utility."""
import re
from os import PathLike
from typing import Tuple, Union

import magic

PATTERN = re.compile(r'(?<=, )(?P<width>\d+)x(?P<height>\d+)')


def get_jpeg_size(filepath: Union[str, PathLike]) -> Tuple[int, int]:
    """Get JPEG image width and height by parsing the file header.
    :param filepath: Path to a JPEG file.
    :return: Image width and height.
    """
    magic_out = magic.from_file(filepath)
    if magic_out.startswith('JPEG image data'):
        out = PATTERN.search(magic_out)
        if out:
            return int(out['width']), int(out['height'])
        raise ValueError('Failed to get image size from JPEG header.')
    raise ValueError('Not a JPEG file')
