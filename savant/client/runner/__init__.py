from dataclasses import dataclass
from typing import Optional

from savant.client.log_provider import LogProvider, Logs


@dataclass
class LogResult:
    trace_id: Optional[str]
    log_provider: Optional[LogProvider]

    def logs(self) -> Logs:
        if self.log_provider is not None and self.trace_id is not None:
            return self.log_provider.logs(self.trace_id)
        return Logs([])
