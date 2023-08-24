from datetime import datetime
from typing import List, Optional

import requests

from savant.client.log_provider import LogEntry, LogProvider


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

    def _fetch_logs(self, trace_id: str) -> List[LogEntry]:
        # TODO: use gRPC API since HTTP is internal
        # https://www.jaegertracing.io/docs/1.48/apis/#grpcprotobuf-stable
        response = requests.get(f'{self._endpoint}/api/traces/{trace_id}')
        response.raise_for_status()
        logs = []
        for data in response.json()['data']:
            for span in data['spans']:
                for log in span['logs']:
                    timestamp = datetime.fromtimestamp(log['timestamp'] / 1000000)
                    log_entry = LogEntry(timestamp, '', '', '')
                    for field in log['fields']:
                        if field['key'] == 'event':
                            log_entry.message = field['value']
                        elif field['key'] == 'log.level':
                            log_entry.level = field['value']
                        elif field['key'] == 'log.target':
                            log_entry.target = field['value']
                    logs.append(log_entry)

        return logs

    def __repr__(self):
        return (
            f'JaegerLogProvider('
            f'endpoint={self._endpoint}, '
            f'login={self._login}, '
            f'password={"MASKED" if self._password is not None else None})'
        )
