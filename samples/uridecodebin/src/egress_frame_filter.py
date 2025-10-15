from savant_rs.primitives import VideoFrame

from savant.base.frame_filter import BaseFrameFilter


class EgressFilter(BaseFrameFilter):
    """Default ingress filter, filters out frames with no video data."""

    def __init__(self, source_id: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger.info(f'Egress filter initialized for source {source_id}')
        self.source_id = source_id

    def __call__(self, video_frame: VideoFrame) -> bool:
        video_frame.source_id = self.source_id
        return True
