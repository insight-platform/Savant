import os
from datetime import datetime, timedelta
from distutils.util import strtobool
from pathlib import Path

from savant.utils.config import opt_config

TIME_DELTAS = {
    's': lambda x: timedelta(seconds=x),
    'm': lambda x: timedelta(minutes=x),
}


class AwsConfig:
    def __init__(self):
        self.region = os.environ['AWS_REGION']
        self.access_key = os.environ['AWS_ACCESS_KEY']
        self.secret_key = os.environ['AWS_SECRET_KEY']


class FpsMeterConfig:
    def __init__(self):
        self.period_seconds = opt_config('FPS_PERIOD_SECONDS', None, float)
        self.period_frames = opt_config('FPS_PERIOD_FRAMES', 1000, int)
        self.output = opt_config('FPS_OUTPUT', 'stdout')
        assert self.output in [
            'stdout',
            'logger',
        ], 'FPS_OUTPUT must be "stdout" or "logger"'


class Config:
    def __init__(self):
        self.source_id = os.environ['SOURCE_ID']
        self.stream_name = os.environ['STREAM_NAME']
        timestamp = os.environ.get('TIMESTAMP')
        self.timestamp = parse_timestamp(timestamp) if timestamp else datetime.utcnow()

        self.zmq_endpoint = os.environ['ZMQ_ENDPOINT']
        self.sync_output = opt_config('SYNC_OUTPUT', False, strtobool)
        self.playing = opt_config('PLAYING', True, strtobool)
        self.api_port = opt_config('API_PORT', 18367, int)

        self.save_state = opt_config('SAVE_STATE', False, strtobool)
        if self.save_state:
            self.state_path = opt_config('STATE_PATH', Path('state.json'), Path)
        else:
            self.state_path = None

        self.aws: AwsConfig = AwsConfig()
        self.fps_meter: FpsMeterConfig = FpsMeterConfig()


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
