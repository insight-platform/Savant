from collections import deque
from dataclasses import dataclass
from typing import Dict, List, NamedTuple, Optional

import cv2

from savant.deepstream.auxiliary_stream import AuxiliaryStream
from savant.deepstream.meta.frame import NvDsFrameMeta
from savant.deepstream.opencv_utils import draw_rect, nvds_to_gpu_mat
from savant.deepstream.pyfunc import NvDsPyFuncPlugin
from savant.gstreamer import Gst
from savant.utils import log as logging


class Frame(NamedTuple):
    source_id: str
    batch_id: int
    pts: int
    duration: int
    content: cv2.cuda.GpuMat


@dataclass
class Source:
    id: str


def prepare_sources(sources_cfg: List[Dict]) -> List[Source]:
    return [
        Source(
            id=source_cfg['id'],
        )
        for source_cfg in sources_cfg
    ]


class Batch:
    def __init__(self, batch_id: int, source: str, frame: Frame, sources: List[Source]):
        self.batch_id = batch_id
        self.frames: Dict[str, Frame] = {source: frame}
        self.frames[source] = frame
        self.sources = sources

    def add_frame(self, source: str, frame: Frame):
        self.frames[source] = frame

    def get_frames(self) -> Dict[str, Frame]:
        return self.frames

    def is_complete(self) -> bool:
        return len(self.frames) == len(self.sources)


class CombineFrames(NvDsPyFuncPlugin):
    def __init__(
        self,
        source: Dict,
        output: Dict,
        codec_params: Dict,
        queue_size: int = 20,
        report_interval: float = 60,
        **kwargs,
    ):
        self.batches: Dict[int, Batch] = {}
        self.logger = logging.get_logger(f'{__name__}.{self.__class__.__name__}')
        self.output_width, self.output_height = output['width'], output['height']
        self.logger.info(
            f'Output frame dimensions: {self.output_width}x{self.output_height}'
        )

        self.sources: List[Source] = prepare_sources(source['sources'])
        self.source_by_id: Dict[str, Source] = {
            source.id: source for source in self.sources
        }
        for source in self.sources:
            self.logger.info(f'Source {source.id}')

        self.output_source_id = output['source_id']
        self.output_framerate = output['framerate']
        self.codec_params = codec_params

        self.queue_size = queue_size

        self.output_stream: Optional[AuxiliaryStream] = None
        self.last_batch_id: Optional[int] = None
        self.last_pts: Optional[int] = None

        self.logger.info(f'Initializing CombineFrames plugin')
        super().__init__(**kwargs)

    def process_frame(self, buffer: Gst.Buffer, frame_meta: NvDsFrameMeta):
        batch_attr = (
            frame_meta.video_frame.get_attribute('retina-rtsp', 'batch-id')
            .values[0]
            .as_string()
        )
        batch_id = int(batch_attr)
        sources = (
            frame_meta.video_frame.get_attribute('retina-rtsp', 'batch-sources')
            .values[0]
            .as_strings()
        )

        full_batch = True
        if len(sources) != 2:
            full_batch = False

        self.logger.debug(
            f'Batch ID: {batch_id}, Sources: {sources}, Full batch: {full_batch}'
        )

        self.logger.debug(
            'Processing frame %s from source %s and batch %s. Batch %s full.',
            frame_meta.pts,
            frame_meta.source_id,
            batch_id,
            'is' if full_batch else 'is not',
        )

        if not full_batch:
            self.logger.warning(
                'Batch %s %s, Frame %s/%s is not part of a full batch. Skipping.',
                batch_id,
                sources,
                frame_meta.source_id,
                frame_meta.pts,
            )
            return

        with nvds_to_gpu_mat(buffer, frame_meta.frame_meta) as frame_mat:
            frame = Frame(
                source_id=frame_meta.source_id,
                batch_id=batch_id,
                pts=frame_meta.pts,
                duration=frame_meta.duration,
                content=frame_mat.clone(),
            )

            if batch_id not in self.batches:
                self.batches[batch_id] = Batch(
                    batch_id, frame_meta.source_id, frame, self.sources
                )
            else:
                self.batches[batch_id].add_frame(frame_meta.source_id, frame)

            if len(self.batches) == self.queue_size:
                min_batch_id = min(self.batches.keys())
                current_batch = self.batches[min_batch_id]
                del self.batches[current_batch.batch_id]

                self.logger.debug(
                    f'Current batch: {current_batch.batch_id}, is complete: {current_batch.is_complete()}'
                )

                if not current_batch.is_complete():
                    self.logger.warning(
                        f'Batch {current_batch.batch_id} is not complete. Skipping.'
                    )
                    return

                self.logger.debug(f'Batch {min_batch_id} is complete. Processing.')
                if self.last_batch_id:
                    if min_batch_id <= self.last_batch_id:
                        self.logger.warning(
                            f'Batch {min_batch_id} is not greater than last batch {self.last_batch_id}. Skipping.'
                        )
                        return

                self.last_batch_id = min_batch_id

                frames_to_combine = current_batch.get_frames()
                stream = self.get_cuda_stream(frame_meta)
                output_buffer = self.create_output_buffer(frames_to_combine)
                if not output_buffer:
                    return

                with nvds_to_gpu_mat(output_buffer, batch_id=0) as output_mat:
                    self.combine_frames(frames_to_combine, output_mat, stream)

    def create_output_buffer(
        self, frames_to_combine: Dict[str, Frame]
    ) -> Optional[Gst.Buffer]:
        batch_id = next(iter(frames_to_combine.values())).batch_id
        pts, duration = next(
            (frames_to_combine[x.id].pts, frames_to_combine[x.id].duration)
            for x in self.sources
            if x.id in frames_to_combine
        )
        if self.last_pts and pts <= self.last_pts:
            self.logger.warning(
                f'Batch {batch_id} has PTS {pts} which is not greater than last PTS {self.last_pts}. Skipping.'
            )
            return None
        self.last_pts = pts

        self.logger.debug(
            'Dewarping and combining %s frames from batch %s. PTS: %s.',
            len(frames_to_combine),
            batch_id,
            pts,
        )
        _, output_buffer = self.output_stream.create_frame(pts, duration)

        return output_buffer

    def combine_frames(
        self,
        frames_to_combine: Dict[str, Frame],
        output_mat: cv2.cuda.GpuMat,
        stream: Optional[cv2.cuda.Stream],
    ):
        batch_id = next(iter(frames_to_combine.values())).batch_id
        self.logger.debug('[Batch %s] Combining frames.', batch_id)

        for i, source in enumerate(self.sources):
            self.logger.debug('[Batch %s] Processing source %s.', batch_id, source.id)
            frame = frames_to_combine.get(source.id)
            if frame is None:
                self.logger.warning(
                    '[Batch %s] Missing frame for source %s.', batch_id, source.id
                )
                return

        stitched_width, stitched_height = output_mat.size()
        main_w, main_h = frames_to_combine['main'].content.size()
        overlay_w, overlay_h = frames_to_combine['overlay'].content.size()

        assert main_w == overlay_w
        assert main_h == overlay_h
        assert main_w == stitched_width
        assert main_h == stitched_height
        assert main_w == self.output_width

        main_cam = self.source_by_id['main']
        main_frame: Frame = frames_to_combine.get(main_cam.id)
        main_frame_content: cv2.cuda.GpuMat = main_frame.content

        overlay_cam = self.source_by_id['overlay']
        overlay_frame: Frame = frames_to_combine.get(overlay_cam.id)
        overlay_frame_content: cv2.cuda.GpuMat = overlay_frame.content

        main_crop_area = (0, 0, self.output_width, self.output_height)

        overlay_crop_area = (
            self.output_width // 2,
            0,
            self.output_width // 2,
            self.output_height // 2,
        )

        main_crop = cv2.cuda.GpuMat(main_frame_content, main_crop_area)
        main_crop_dest = cv2.cuda.GpuMat(output_mat, main_crop_area)
        main_crop.copyTo(dst=main_crop_dest, stream=stream)

        overlay_crop = cv2.cuda.GpuMat(overlay_frame_content, overlay_crop_area)
        overlay_crop_dest = cv2.cuda.GpuMat(output_mat, overlay_crop_area)
        overlay_crop.copyTo(dst=overlay_crop_dest, stream=stream)

        stream.waitForCompletion()
        draw_rect(
            output_mat,
            (self.output_width // 2, 0, self.output_width, self.output_height // 2),
            (255, 255, 255, 255),
            2,
        )

        self.logger.debug('[Batch %s] Frames combined.', batch_id)

    def on_start(self) -> bool:
        self.logger.info('Starting CombineFrames plugin')
        if not super().on_start():
            return False

        try:
            self.output_stream = self.auxiliary_stream(
                source_id=self.output_source_id,
                width=self.output_width,
                height=self.output_height,
                codec_params=self.codec_params,
                framerate=self.output_framerate,
            )
        except Exception as e:
            self.logger.error('Failed to create output stream: %s', e, exc_info=True)
            return False

        return True

    def on_source_eos(self, source_id: str):
        self.logger.info('Got EOS from source %s.', source_id)
        for source in self.sources:
            source.pending_frames = deque()

    def on_stop(self) -> bool:
        self.output_stream = None
        return super().on_stop()
