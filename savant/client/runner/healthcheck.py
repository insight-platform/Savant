import time
from http import HTTPStatus
from typing import List, Optional

import requests
from requests import RequestException

from savant.healthcheck.status import PipelineStatus
from savant.utils.logging import get_logger

logger = get_logger(__name__)


class HealthCheck:
    """Service to check the health of the module.

    :param url: URL of the health check endpoint.
    :param interval: Interval between health checks in seconds.
    :param timeout: Timeout for waiting the module to be ready in seconds.
    :param ready_statuses: List of statuses that indicate the module is ready.
    """

    def __init__(
        self,
        url: str,
        interval: float,
        timeout: float,
        ready_statuses: List[PipelineStatus],
    ):
        self._url = url
        self._check_interval = interval
        self._wait_timeout = timeout
        self._ready_statuses = ready_statuses
        self._last_check_ts = 0
        self._last_status = None

    def check(self) -> Optional[PipelineStatus]:
        """Check the health of the module."""

        logger.debug('Checking module status.')
        try:
            response = requests.get(self._url)
        except RequestException as e:
            logger.warning('Health check failed. Error: %s.', e)
            return None

        if response.status_code not in [HTTPStatus.OK, HTTPStatus.SERVICE_UNAVAILABLE]:
            # Only OK and SERVICE_UNAVAILABLE status codes are expected.
            raise RuntimeError(
                f'Health check failed. Status code: {response.status_code}.'
            )

        status = response.text.strip()
        if not status:
            logger.debug('Module has not status yet.')
            return None

        logger.debug('Module status: %s.', status)
        try:
            return PipelineStatus(status)
        except ValueError:
            logger.warning('Unknown status: %s.', status)
            return None

    def wait_module_is_ready(self):
        """Wait until the module is ready."""

        if time.time() - self._last_check_ts >= self._check_interval:
            self._last_status = self.check()
            self._last_check_ts = time.time()

        time_limit = time.time() + self._wait_timeout
        while self._last_status not in self._ready_statuses:
            if time.time() > time_limit:
                raise TimeoutError(
                    f'Module is not ready after {self._wait_timeout} seconds.'
                )
            time.sleep(self._check_interval)
            self._last_status = self.check()
            self._last_check_ts = time.time()
