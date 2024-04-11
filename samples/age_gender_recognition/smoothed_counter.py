"""Smoothed counter module."""

import queue
from collections import defaultdict
from typing import Any, Dict


class SmoothedCounter:
    """Counter that provides smoothing of values over time.

    :param history_len: length of history
    :param smoothed_type: type of smoothing. Available types: mean, median, vote
    :param lifetime: lifetime(in frame) of queue
    """

    def __init__(
        self, history_len: int = 24, smoothed_type: str = 'mean', lifetime: int = 24
    ) -> None:
        self.queues: Dict[queue.Queue] = defaultdict(
            lambda: dict(last_frame_num=0, values=queue.Queue(history_len))
        )
        self.smoothed_type = smoothed_type
        self.lifetime = lifetime

    def get_value(
        self,
        new_value: Any,
        frame_num: int,
        key: Any = 'default',
    ) -> Any:
        """Receive smooth value using new measurement value
        :param new_value: the value of the new measurement.
        :param frame_num: frame number
        :param key: key for queue
        :return: smooth value
        """
        if self.queues[key]['values'].full():
            self.queues[key]['values'].get()
        self.queues[key]['values'].put(new_value)
        self.queues[key]['last_frame_num'] = frame_num
        if self.smoothed_type == 'mean':
            return self._mean_smooth(key)
        elif self.smoothed_type == 'median':
            return self._median_smooth(key)
        elif self.smoothed_type == 'vote':
            return self.vote_smooth(key)
        raise ValueError(f'Unknown smoothed type: {self.smoothed_type}')

    def _mean_smooth(self, external_key: Any) -> Any:
        """Calculate mean value of queue"""
        return sum(self.queues[external_key]['values'].queue) / len(
            self.queues[external_key]['values'].queue
        )

    def _median_smooth(self, external_key: Any) -> Any:
        """Calculate median value of queue"""
        half_index = len(self.queues[external_key]['values'].queue) // 2
        return sorted(self.queues[external_key]['values'].queue)[half_index]

    def vote_smooth(self, external_key: Any) -> Any:
        """Calculate vote value of queue"""
        return max(
            set(self.queues[external_key]['values'].queue),
            key=self.queues[external_key]['values'].queue.count,
        )

    def clean(self, frame_num: int):
        """Clean queues with expired time"""
        keys = list(self.queues.keys())
        for key in keys:
            if frame_num - self.queues[key]['last_frame_num'] > self.lifetime:
                del self.queues[key]
