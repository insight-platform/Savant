import logging
import os
import re
from distutils.util import strtobool
from typing import Optional


def check_display_env(logger: logging.Logger, unset: Optional[bool] = None):
    """Checks if the DISPLAY env var is set,
    if it is set and `UNSET_DISPLAY` is specified,
    it will be unset due to DeepStream limitation:
    "Running a DeepStream application over SSH (via putty)
    with X11 forwarding does not work."
    """
    display = os.environ.get('DISPLAY')
    if not display:
        return

    logger.info(f'DISPLAY env is set to "{display}"')
    if unset is None:
        unset = strtobool(os.environ.get('UNSET_DISPLAY', 'True'))
    if not unset:
        return

    display_pattern = re.compile(
        r'^(?P<hostname>.*):(?P<display_num>\d+).?(?P<screen_num>\d?)$'
    )
    match = display_pattern.match(display)
    if match and match['hostname']:
        logger.warning(
            'Due to DeepStream limitation, DISPLAY env has been reset. '
            'To change this behavior use env UNSET_DISPLAY=False.'
        )
        del os.environ['DISPLAY']
