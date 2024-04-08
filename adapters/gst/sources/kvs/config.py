import os
from datetime import datetime, timedelta
from distutils.util import strtobool
from pathlib import Path

TIME_DELTAS = {
    's': lambda x: timedelta(seconds=x),
    'm': lambda x: timedelta(minutes=x),
}


class Config:
    def __init__(self):
        self.source_id = os.environ['SOURCE_ID']
        self.stream_name = os.environ['STREAM_NAME']
        timestamp = os.environ.get('TIMESTAMP')
        self.timestamp = parse_timestamp(timestamp) if timestamp else datetime.utcnow()
        self.aws_region = os.environ['AWS_REGION']
        self.access_key = os.environ['AWS_ACCESS_KEY']
        self.secret_key = os.environ['AWS_SECRET_KEY']

        self.zmq_endpoint = os.environ['ZMQ_ENDPOINT']
        self.sync_output = bool(strtobool(os.environ.get('SYNC_OUTPUT', 'False')))
        self.api_port = int(os.environ.get('API_PORT', 18367))

        self.save_state = bool(strtobool(os.environ.get('SAVE_STATE', 'False')))
        if self.save_state:
            self.state_path = Path(os.environ.get('STATE_PATH', 'state.json'))
        else:
            self.state_path = None


def parse_timestamp(ts: str) -> datetime:
    """Parse a timestamp string into a datetime object.

    The timestamp can be in the format "YYYY-MM-DDTHH:MM:SS" or a relative time
    in the format "-N[s|m]" where N is an integer and "s" or "m" is the unit of
    time (seconds or minutes)."""

    try:
        return datetime.strptime(ts, '%Y-%m-%dT%H:%M:%S')
    except ValueError:
        unit = ts[-1]
        delta = int(ts[:-1])
        if delta > 0:
            raise ValueError(f'Invalid timestamp: {ts}')
        return datetime.utcnow() + TIME_DELTAS[unit](delta)
