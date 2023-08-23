from typing import Optional

from savant.source_sink_framework.log_provider import LogProvider


class JaegerLogProvider(LogProvider):
    def __init__(
        self,
        endpoint: str,
        login: Optional[str] = None,
        password: Optional[str] = None,
    ):
        self._endpoint = endpoint
        self._login = login
        self._password = password

    def __repr__(self):
        return (
            f'JaegerLogProvider('
            f'endpoint={self._endpoint}, '
            f'login={self._login}, '
            f'password={"MASKED" if self._password is not None else None})'
        )
