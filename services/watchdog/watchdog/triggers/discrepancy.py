"""Discrepancy trigger: detect when egress is idle but ingress is active.

This indicates a stuck module — frames are going in but not coming out.

Metrics are fetched automatically by the watchdog from the ``buffer``
field of the watch config and passed to ``__call__`` as a dict.

Example configuration:

.. code-block:: yaml

    watch:
      - buffer: ${oc.env:BUFFER_URL}
        pyfunc:
          action: restart
          cooldown: 600s
          polling_interval: 10s
          container:
            - labels: [com.savant.module=detector]
          module: watchdog.triggers.discrepancy
          class_name: DiscrepancyCheck
          kwargs:
            egress_idle: 60
            ingress_idle: 30
"""

import logging
import time
from typing import Dict

logger = logging.getLogger(__name__)

LAST_SENT_MESSAGE_METRIC = 'last_sent_message'
LAST_RECEIVED_MESSAGE_METRIC = 'last_received_message'


class DiscrepancyCheck:
    """Trigger when egress is idle AND ingress is active (module stuck).

    :param egress_idle: Seconds of egress idle time before considering it stalled.
    :param ingress_idle: Seconds — ingress must have been active within this
        window for the trigger to fire. If ingress is also idle, the
        problem is upstream (no input), not a stuck module.
    """

    def __init__(self, egress_idle: float, ingress_idle: float):
        self.egress_idle = egress_idle
        self.ingress_idle = ingress_idle

    def __call__(self, metrics: Dict[str, float]) -> bool:
        last_sent = metrics[LAST_SENT_MESSAGE_METRIC]
        last_received = metrics[LAST_RECEIVED_MESSAGE_METRIC]

        now = time.time()
        egress_idle_duration = now - last_sent
        ingress_idle_duration = now - last_received

        egress_is_idle = egress_idle_duration > self.egress_idle
        ingress_is_active = ingress_idle_duration <= self.ingress_idle

        if egress_is_idle and ingress_is_active:
            logger.info(
                'Discrepancy check: egress idle=%.1fs > threshold=%ss, '
                'ingress idle=%.1fs <= threshold=%ss, triggering action',
                egress_idle_duration,
                self.egress_idle,
                ingress_idle_duration,
                self.ingress_idle,
            )
            return True

        logger.debug(
            'Discrepancy check: egress idle=%.1fs (threshold=%ss), '
            'ingress idle=%.1fs (threshold=%ss), no action',
            egress_idle_duration,
            self.egress_idle,
            ingress_idle_duration,
            self.ingress_idle,
        )
        return False
