"""Smoothed counter module."""
from typing import Any


class SmoothedCounter:
    """Counter that provides median of values encountered during
    previous update period.

    :param update_period: time in seconds, how often should the counter update.
    """

    def __init__(self, update_period: float = 1) -> None:
        self.frame_period_ns = update_period * 10**9
        self.last_change_time = 0
        self.current_val = 0
        self.values = []

    def get_value(self, time: int, new_value: Any) -> Any:
        """Receive new measurement value, update smoothed value if necessary,
        return median value of the previous update period.

        :param time: new measurement's time in nanoseconds.
        :new_value: the value of the new measurement.
        :return: median value of the previous update period.
        """
        self.values.append(new_value)

        if time - self.last_change_time >= self.frame_period_ns:
            values = sorted(self.values)
            self.current_val = values[len(values) // 2]
            self.values = []
            self.last_change_time = time

        return self.current_val
