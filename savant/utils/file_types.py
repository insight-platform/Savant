import subprocess
from enum import Enum
from pathlib import Path
from typing import List, Optional, Tuple


class FileType(Enum):
    VIDEO = 'video'
    PICTURE = 'picture'

    @staticmethod
    def from_mime_type(mime_type: Optional[str]) -> Optional['FileType']:
        if mime_type is None:
            return None
        if mime_type.startswith('video/'):
            return FileType.VIDEO
        if mime_type.startswith('image/'):
            return FileType.PICTURE


def parse_mime_types(files: List[Path]) -> List[Tuple[Path, str]]:
    output = subprocess.check_output(
        ['file', '--no-pad', '--mime-type'] + [str(x) for x in files]
    )
    mime_types = []
    for line in output.decode().strip().split('\n'):
        path, mime_type = line.rsplit(': ', 1)
        mime_types.append((Path(path), mime_type))

    return mime_types
