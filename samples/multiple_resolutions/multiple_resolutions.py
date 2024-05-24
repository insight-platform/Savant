from typing import Dict, List, NamedTuple

import cv2

from savant.deepstream.auxiliary_stream import AuxiliaryStream
from savant.deepstream.meta.frame import NvDsFrameMeta
from savant.deepstream.opencv_utils import nvds_to_gpu_mat
from savant.deepstream.pyfunc import NvDsPyFuncPlugin
from savant.gstreamer import Gst


class ResolutionDesc(NamedTuple):
    suffix: str
    width: int
    height: int


class MultipleResolutions(NvDsPyFuncPlugin):
    def __init__(
        self,
        resolutions: List[Dict],
        codec_params: Dict,
        **kwargs,
    ):
        self.resolutions = [
            ResolutionDesc(suffix=x['suffix'], width=x['width'], height=x['height'])
            for x in resolutions
        ]
        self.codec_params = codec_params

        # source_id -> resolution -> AuxiliaryStream
        self.aux_streams: Dict[str, Dict[str, AuxiliaryStream]] = {}

        super().__init__(**kwargs)

    def on_source_add(self, source_id: str):
        self.logger.info('Source %s added.', source_id)
        if source_id in self.aux_streams:
            self.logger.info('Source %s already has auxiliary streams.', source_id)
            return

        self.aux_streams[source_id] = {}
        for resolution in self.resolutions:
            aux_source_id = f'{source_id}{resolution.suffix}'
            self.logger.info(
                'Creating auxiliary stream %s for source %s.',
                aux_source_id,
                source_id,
            )
            aux_stream = self.auxiliary_stream(
                source_id=aux_source_id,
                width=resolution.width,
                height=resolution.height,
                codec_params=self.codec_params,
            )
            self.aux_streams[source_id][resolution.suffix] = aux_stream

    def process_frame(self, buffer: Gst.Buffer, frame_meta: NvDsFrameMeta):
        stream = self.get_cuda_stream(frame_meta)
        with nvds_to_gpu_mat(buffer, frame_meta.frame_meta) as frame_mat:
            for resolution in self.resolutions:
                aux_stream = self.aux_streams[frame_meta.source_id][resolution.suffix]
                aux_frame, aux_buffer = aux_stream.create_frame(
                    pts=frame_meta.pts,
                    duration=frame_meta.duration,
                )
                with nvds_to_gpu_mat(aux_buffer, batch_id=0) as aux_mat:
                    cv2.cuda.resize(
                        src=frame_mat,
                        dst=aux_mat,
                        dsize=(resolution.width, resolution.height),
                        stream=stream,
                    )

    def on_stop(self) -> bool:
        self.aux_streams = {}
        return super().on_stop()

    def on_source_eos(self, source_id: str):
        self.logger.info('Got EOS from source %s.', source_id)
        if source_id not in self.aux_streams:
            self.logger.info('Source %s does not have auxiliary streams.', source_id)
            return
        for aux_stream in self.aux_streams[source_id].values():
            aux_stream.eos()
