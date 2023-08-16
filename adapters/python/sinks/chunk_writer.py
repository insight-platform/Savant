import math
from typing import List

from savant_rs.primitives import EndOfStream, VideoFrame

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

    def write_video_frame(
        self,
        frame: VideoFrame,
        data,
        can_start_new_chunk: bool,
    ) -> bool:
        if can_start_new_chunk and 0 < self.chunk_size <= self.frames_in_chunk:
            self.close()
        if not self.opened:
            self.open()
        frame_num = self.frames_in_chunk
        result = self._write_video_frame(frame, data, frame_num)
        self.frames_in_chunk += 1
        return result

    def write_eos(self, eos: EndOfStream) -> bool:
        if not self.opened:
            self.open()
        result = self._write_eos(eos)
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

    def _write_video_frame(self, frame: VideoFrame, data, frame_num: int) -> bool:
        pass

    def _write_eos(self, eos: EndOfStream) -> bool:
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

    def _write_video_frame(self, frame: VideoFrame, data, frame_num: int) -> bool:
        for writer in self.writers:
            if not writer._write_video_frame(frame, data, frame_num):
                return False
        return True

    def _write_eos(self, eos: EndOfStream) -> bool:
        for writer in self.writers:
            if not writer._write_eos(eos):
                return False
        return True
