from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class LogEntry:
    """Single log entry."""

    timestamp: datetime
    level: str
    target: str
    message: str
    attributes: Dict[str, Any]
    _pretty_format: Optional[str] = field(init=False, repr=False, default=None)

    def pretty_format(self) -> str:
        """Get pretty formatted string of log entry."""

        if self._pretty_format is None:
            ts = self.timestamp.strftime('%Y-%m-%d %H:%M:%S.%f')
            message = f'[{self.level}] [{self.target}] {self.message}'
            lines = [f'{ts} | {message}']
            for k, v in sorted(self.attributes.items()):
                lines.append(f'{" " * len(ts)} | {k}={v}')
            self._pretty_format = '\n'.join(lines)

        return self._pretty_format


class Logs:
    """Collection of log entries for specific trace ID."""

    def __init__(self, entries: List[LogEntry]):
        self._entries = sorted(entries, key=lambda e: e.timestamp)
        self._pretty_format: Optional[str] = None

    @property
    def entries(self) -> List[LogEntry]:
        """Get list of log entries."""
        return self._entries

    def pretty_format(self) -> str:
        """Get pretty formatted string of logs."""

        if self._pretty_format is None:
            self._pretty_format = '\n'.join(
                entry.pretty_format() for entry in self._entries
            )
        return self._pretty_format

    def pretty_print(self):
        """Print pretty formatted logs."""
        print(self.pretty_format())


class LogProvider(ABC):
    """Interface for log providers."""

    def logs(self, trace_id: str) -> Logs:
        """Fetch logs for given trace ID."""
        return Logs(self._fetch_logs(trace_id))

    @abstractmethod
    def _fetch_logs(self, trace_id: str) -> List[LogEntry]:
        pass
