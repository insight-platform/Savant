import math
from typing import Dict, List, Optional
from savant.utils.logging import get_logger


class ChunkWriter:
    """Writes data in chunks."""

    def __init__(self, chunk_size: int):
        self.logger = get_logger(f'{__name__}.{self.__class__.__name__}')
        self.chunk_size = chunk_size
        if chunk_size > 0:
            self.chunk_size_digits = int(math.log10(chunk_size)) + 1
        else:
            self.chunk_size_digits = 6
        self.chunk_idx = -1
        self.frames_in_chunk = 0
        self.opened = False

    def write(
        self,
        message,
        data,
        can_start_new_chunk: bool,
        is_frame: bool = True,
    ) -> bool:
        if can_start_new_chunk and 0 < self.chunk_size <= self.frames_in_chunk:
            self.close()
        if not self.opened:
            self.open()
        frame_num = self.frames_in_chunk if is_frame else None
        result = self._write(message, data, frame_num)
        if is_frame:
            self.frames_in_chunk += 1
        return result

    def open(self):
        if self.opened:
            return
        self.chunk_idx += 1
        self.frames_in_chunk = 0
        self._open()
        self.opened = True

    def close(self):
        self.flush()
        if not self.opened:
            return
        self._close()
        self.opened = False

    def flush(self):
        if not self.opened:
            return
        self._flush()

    def _open(self):
        pass

    def _close(self):
        pass

    def _flush(self):
        pass

    def _write(self, message, data, frame_num: Optional[int]) -> bool:
        pass


class CompositeChunkWriter(ChunkWriter):
    def __init__(self, writers: List[ChunkWriter], chunk_size: int):
        self.writers = writers
        super().__init__(chunk_size)

    def _open(self):
        for writer in self.writers:
            writer.open()

    def _close(self):
        for writer in self.writers:
            writer.close()

    def _flush(self):
        for writer in self.writers:
            writer.flush()

    def _write(
        self,
        message: Dict,
        data: List[bytes],
        frame_num: Optional[int],
    ) -> bool:
        for writer in self.writers:
            if not writer._write(message, data, frame_num):
                return False
        return True
