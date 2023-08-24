from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import List


@dataclass
class LogEntry:
    timestamp: datetime
    level: str
    target: str
    message: str


class Logs:
    def __init__(self, entries: List[LogEntry]):
        self._entries = entries

    @property
    def entries(self) -> List[LogEntry]:
        return self._entries

    def pretty_print(self):
        # TODO:
        # ts | log message
        #    | attributes
        for entry in self._entries:
            print(f"{entry.timestamp} [{entry.level}] [{entry.target}] {entry.message}")


class LogProvider(ABC):
    def logs(self, trace_id: str) -> Logs:
        return Logs(self._fetch_logs(trace_id))

    @abstractmethod
    def _fetch_logs(self, trace_id: str) -> List[LogEntry]:
        pass
