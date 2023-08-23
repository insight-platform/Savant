from savant.source_sink_framework.builder.sink import SinkBuilder
from savant.source_sink_framework.builder.source import SourceBuilder
from savant.source_sink_framework.log_provider import LogProvider
from savant.source_sink_framework.runner.source_sink import SourceSinkRunner


class SourceSinkBuilder:
    def __init__(
        self,
        source: SourceBuilder = None,
        sink: SinkBuilder = None,
    ):
        self._source = source if source is not None else SourceBuilder()
        self._sink = sink if sink is not None else SinkBuilder()

    def with_timeout(self, timeout: float) -> 'SourceSinkBuilder':
        return SourceSinkBuilder(
            source=self._source.with_timeout(timeout),
            sink=self._sink,
        )

    def with_sockets(self, source: str, sink: str) -> 'SourceSinkBuilder':
        return SourceSinkBuilder(
            source=self._source.with_socket(source),
            sink=self._sink.with_socket(sink),
        )

    def with_log_provider(self, log_provider: LogProvider) -> 'SourceSinkBuilder':
        return SourceSinkBuilder(
            source=self._source.with_log_provider(log_provider),
            sink=self._sink.with_log_provider(log_provider),
        )

    def build(self) -> SourceSinkRunner:
        return SourceSinkRunner(
            source=self._source.build(),
            sink=self._sink.build(),
        )

    def __repr__(self):
        return f'SourceSinkBuilder(source={self._source}, sink={self._sink})'
