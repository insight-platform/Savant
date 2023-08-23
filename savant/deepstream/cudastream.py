import cv2

from savant.deepstream.meta.frame import NvDsFrameMeta
from savant.utils.logging import LoggerMixin


class CudaStreams(LoggerMixin):
    """Class for managing CUDA streams for asynchronous frame processing."""

    def __init__(self):
        super().__init__()
        self.frame_streams = {}

    def get_cuda_stream(self, frame_meta: NvDsFrameMeta):
        """Get a CUDA stream that can be used to
        asynchronously process a frame in a batch.
        """
        self.logger.debug(
            'Getting CUDA stream for frame with batch_id=%d', frame_meta.batch_id
        )
        if frame_meta.batch_id not in self.frame_streams:
            self.logger.debug(
                'No existing CUDA stream for frame with batch_id=%d, init new',
                frame_meta.batch_id,
            )
            self.frame_streams[frame_meta.batch_id] = cv2.cuda.Stream()

        return self.frame_streams[frame_meta.batch_id]

    def sync_cuda_streams(self):
        """
        Wait for all CUDA streams to complete.
        :return:
        """
        for stream in self.frame_streams.values():
            stream.waitForCompletion()
        self.frame_streams.clear()

    def __enter__(self):
        return self

    def __exit__(self, *exc_details):
        self.sync_cuda_streams()
