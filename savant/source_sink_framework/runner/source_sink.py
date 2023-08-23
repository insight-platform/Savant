from dataclasses import dataclass, field
from typing import Iterable, Optional

from savant_rs.primitives import EndOfStream, VideoFrame

from savant.source_sink_framework.frame_source import FrameSource
from savant.source_sink_framework.runner.sink import SinkResult, SinkRunner
from savant.source_sink_framework.runner.source import SourceResult, SourceRunner


@dataclass
class SourceSinkResult:
    status: str
    frame_meta: Optional[VideoFrame]
    frame_content: Optional[bytes] = field(repr=False)
    eos: Optional[EndOfStream]

    @staticmethod
    def build(
        source_result: SourceResult,
        sink_result: SinkResult,
    ) -> 'SourceSinkResult':
        return SourceSinkResult(
            status=source_result.status,
            frame_meta=sink_result.frame_meta,
            frame_content=sink_result.frame_content,
            eos=sink_result.eos,
        )


class SourceSinkRunner:
    def __init__(
        self,
        source: SourceRunner,
        sink: SinkRunner,
    ):
        self._source = source
        self._sink = sink

    def __call__(self, source: FrameSource) -> SourceSinkResult:
        return self.send(source)

    def send(self, source: FrameSource, send_eos: bool = True) -> SourceSinkResult:
        source_result = self._source.send(source, send_eos)
        sink_result = next(self._sink)
        if send_eos and sink_result.eos is None:
            next(self._sink)
        return SourceSinkResult.build(source_result, sink_result)

    def send_iter(
        self,
        sources: Iterable[FrameSource],
        send_eos: bool = True,
    ) -> Iterable[SourceSinkResult]:
        # TODO: send and wait results in different threads
        sink_result = None
        for source_result in self._source.send_iter(sources, send_eos):
            sink_result = next(self._sink)
            if sink_result.eos is not None:
                sink_result = next(self._sink)
            yield SourceSinkResult.build(source_result, sink_result)

        if sink_result is not None and sink_result.eos is None:
            next(self._sink)
