import time
from http import HTTPStatus
from typing import List, Optional

import requests
from requests import JSONDecodeError, RequestException

from savant.utils.logging import get_logger

logger = get_logger(__name__)


class HealthCheck:
    """Service to check the health of the module.

    :param url: URL of the health check endpoint.
    :param interval: Interval between health checks in seconds.
    :param timeout: Timeout for waiting the module to be ready in seconds.
    """

    def __init__(
        self,
        url: str,
        interval: float,
        timeout: float,
    ):
        self._url = url
        self._check_interval = interval
        self._wait_timeout = timeout
        self._last_check_ts = 0
        self._last_status = None

    def check(self) -> Optional[str]:
        """Check the health of the module."""

        logger.debug('Checking module status.')
        try:
            response = requests.get(self._url)
        except RequestException as e:
            logger.warning('Health check failed. Error: %s.', e)
            return None

        if response.status_code != HTTPStatus.OK:
            logger.warning(
                'Health check failed (Expected HTTP 200 OK): unexpected HTTP status code: %s.',
                response.status_code,
            )
            return None

        try:
            status = response.json()
        except JSONDecodeError:
            logger.warning(
                'Failed to decode JSON status. Raw health check status: %s',
                response.text,
            )
            status = None

        if not status:
            logger.debug('Module has no status yet.')
            return None

        logger.debug('Module status: %s.', status)
        return status

    def wait_module_is_ready(self):
        """Wait until the module is ready."""

        if time.time() - self._last_check_ts >= self._check_interval:
            self._last_status = self.check()
            self._last_check_ts = time.time()

        time_limit = time.time() + self._wait_timeout
        while self._last_status != 'running':
            if time.time() > time_limit:
                raise TimeoutError(
                    f'Module is not ready after {self._wait_timeout} seconds.'
                )
            time.sleep(self._check_interval)
            self._last_status = self.check()
            self._last_check_ts = time.time()
