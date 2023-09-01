from datetime import datetime
from typing import List, Optional

import requests

from savant.client.log_provider import LogEntry, LogProvider


class JaegerLogProvider(LogProvider):
    """Log provider for Jaeger.

    :param endpoint: Jaeger endpoint URL.
    :param login: Jaeger login.
    :param password: Jaeger password.
    """

    def __init__(
        self,
        endpoint: str,
        login: Optional[str] = None,
        password: Optional[str] = None,
    ):
        self._endpoint = endpoint
        self._login = login
        self._password = password

    def _fetch_logs(self, trace_id: str) -> List[LogEntry]:
        # TODO: use gRPC API since HTTP is internal
        # https://www.jaegertracing.io/docs/1.48/apis/#grpcprotobuf-stable
        response = requests.get(f'{self._endpoint}/api/traces/{trace_id}')
        response.raise_for_status()
        logs = []
        for data in response.json()['data']:
            for span in data['spans']:
                for log in span['logs']:
                    timestamp = datetime.fromtimestamp(log['timestamp'] / 1_000_000)
                    fields = {field['key']: field['value'] for field in log['fields']}
                    if fields.get('event.name') != 'log-record':
                        continue
                    try:
                        entry = LogEntry(
                            timestamp=timestamp,
                            level=fields['log.level'],
                            target=fields['log.target'],
                            message=fields['event'],
                            attributes={
                                k: v
                                for k, v in fields.items()
                                if k
                                not in [
                                    'event',
                                    'event.domain',
                                    'event.name',
                                    'log.level',
                                    'log.target',
                                ]
                            },
                        )
                    except KeyError:
                        continue
                    logs.append(entry)

        return logs

    def __repr__(self):
        return (
            f'JaegerLogProvider('
            f'endpoint={self._endpoint}, '
            f'login={self._login}, '
            f'password={"MASKED" if self._password is not None else None})'
        )
