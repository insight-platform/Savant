"""Discrepancy trigger: detect when egress is idle but ingress is active.

This indicates a stuck module — frames are going in but not coming out.

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

The ``buffer_url`` is passed automatically by the watchdog from the
``buffer`` field of the watch config.
"""

import logging
import time

from watchdog.buffer_metrics import get_metrics, parse_metrics

logger = logging.getLogger('PipelineWatchdog')

LAST_SENT_MESSAGE_METRIC = 'last_sent_message'
LAST_RECEIVED_MESSAGE_METRIC = 'last_received_message'


class DiscrepancyCheck:
    """Trigger when egress is idle AND ingress is active (module stuck).

    Args:
        buffer_url: Buffer URL to fetch metrics from. Passed automatically
            by the watchdog from the ``buffer`` field of the watch config.
        egress_idle: Seconds of egress idle time before considering it stalled.
        ingress_idle: Seconds — ingress must have been active within this
            window for the trigger to fire. If ingress is also idle, the
            problem is upstream (no input), not a stuck module.
    """

    def __init__(self, buffer_url: str, egress_idle: float, ingress_idle: float):
        self.buffer_url = buffer_url
        self.egress_idle = egress_idle
        self.ingress_idle = ingress_idle

    async def __call__(self) -> bool:
        content = await get_metrics(self.buffer_url)
        metrics = await parse_metrics(content)

        now = time.time()

        last_sent = metrics.get(LAST_SENT_MESSAGE_METRIC)
        last_received = metrics.get(LAST_RECEIVED_MESSAGE_METRIC)

        if last_sent is None or last_received is None:
            logger.warning(
                'DiscrepancyCheck [%s]: missing metric(s) '
                '(last_sent_message=%s, last_received_message=%s), skipping',
                self.buffer_url,
                last_sent,
                last_received,
            )
            return False

        egress_idle_duration = now - last_sent
        ingress_idle_duration = now - last_received

        egress_is_idle = egress_idle_duration > self.egress_idle
        ingress_is_active = ingress_idle_duration <= self.ingress_idle

        if egress_is_idle and ingress_is_active:
            logger.info(
                'DiscrepancyCheck [%s]: module stuck — '
                'egress idle %.1fs (threshold %ss), '
                'ingress active %.1fs (threshold %ss)',
                self.buffer_url,
                egress_idle_duration,
                self.egress_idle,
                ingress_idle_duration,
                self.ingress_idle,
            )
            return True

        logger.debug(
            'DiscrepancyCheck [%s]: healthy — egress idle %.1fs, ingress idle %.1fs',
            self.buffer_url,
            egress_idle_duration,
            ingress_idle_duration,
        )
        return False
