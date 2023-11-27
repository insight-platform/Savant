from abc import ABC, abstractmethod
from threading import Thread
from typing import Optional

from savant.utils.logging import get_logger


class BaseThreadWorker(ABC):
    thread: Thread

    def __init__(
        self,
        thread_name: str,
        logger_name: Optional[str] = None,
        daemon: bool = False,
    ):
        self.logger = get_logger(
            logger_name
            if logger_name is not None
            else f'{self.__class__.__module__}.{self.__class__.__name__}'
        )
        self.thread_name = thread_name
        self.is_running = False
        self.error: Optional[str] = None
        self.daemon = daemon

    def start(self):
        self.is_running = True
        self.thread = Thread(
            name=self.thread_name,
            target=self.workload,
            daemon=self.daemon,
        )
        self.thread.start()

    def stop(self):
        self.is_running = False

    def join(self, timeout=None):
        self.thread.join(timeout)

    @abstractmethod
    def workload(self):
        pass

    def set_error(self, error: str):
        """Log and set the error message if the adapter failed to run.

        Only sets the first error.
        """

        self.logger.error(error, exc_info=True)
        if self.error is None:
            self.error = error
