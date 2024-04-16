import time
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Optional

from adapters.shared.thread import BaseThreadWorker

from . import LOGGER_PREFIX
from .stream_model import StreamModel

STATE_SAVE_INTERVAL = 1


class State(BaseThreadWorker):
    """State manager.

    Keeps track of the state of the stream and saves it to a file.
    """

    def __init__(self, path: Path, state_save_interval: float = STATE_SAVE_INTERVAL):
        super().__init__('State', logger_name=f'{LOGGER_PREFIX}.state')
        if path.exists() and not path.is_file():
            raise ValueError(f'{path} is not a file')
        if not path.parent.exists():
            path.parent.mkdir(parents=True)
        self.path = path
        self.state_save_interval = state_save_interval
        self.state: Optional[StreamModel] = self.load()
        self.updated = False
        self.lock = Lock()

    def load(self):
        """Load state from file."""

        if not self.path.exists():
            return None
        self.logger.info('Loading state from %s', self.path)
        with self.path.open('r') as f:
            return StreamModel.parse_raw(f.read())

    def save(self):
        """Save state to file."""

        if self.state is None:
            return
        self.logger.debug('Saving state to %s', self.path)
        with self.path.open('w') as f:
            f.write(self.state.json())

    def update(
        self,
        state: Optional[StreamModel] = None,
        last_ts: Optional[float] = None,
    ):
        """Update state."""

        with self.lock:
            if state is not None:
                self.state = state.copy()
            if last_ts is not None:
                self.state.timestamp = datetime.utcfromtimestamp(last_ts)
            self.logger.debug('State updated to %r', self.state)
            self.updated = True

    def workload(self):
        """Periodically save state to file."""

        while self.is_running:
            if self.updated:
                self.save()
                self.updated = False
            time.sleep(self.state_save_interval)
        self.save()

    def get_state(self):
        """Get current state."""

        with self.lock:
            if self.state is not None:
                return self.state.copy()
