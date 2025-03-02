from dataclasses import dataclass
from typing import Optional

from savant.client.log_provider import LogProvider, Logs


@dataclass
class LogResult:
    trace_id: Optional[str]
    """OpenTelemetry trace ID of the message."""
    log_provider: Optional[LogProvider]
    """Log provider for to fetch the logs."""

    def logs(self) -> Logs:
        """Fetch logs from log provider for this result."""

        if self.log_provider is not None and self.trace_id is not None:
            return self.log_provider.logs(self.trace_id)
        return Logs([])
