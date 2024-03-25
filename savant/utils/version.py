"""Parses VERSION file."""
from pathlib import Path
from typing import List

from savant.utils.singleton import SingletonMeta

VERSION_FILE_PATH = Path(__file__).parent.parent / 'VERSION'


__all__ = ['version']


class Version(metaclass=SingletonMeta):
    __slots__ = '_versions'

    def __init__(self, version_file_path: str = VERSION_FILE_PATH):
        with open(version_file_path, 'r') as file_obj:
            self._versions = dict(
                [
                    map(lambda s: s.strip(), line.split('=', 2))
                    for line in file_obj.read().splitlines()
                ]
            )

    @property
    def SAVANT(self):
        return self._versions['SAVANT']

    @property
    def SAVANT_RS(self):
        return self._versions['SAVANT_RS']

    @property
    def DEEPSTREAM(self):
        return self._versions['DEEPSTREAM']


version = Version()
