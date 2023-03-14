class SmoothedCounter:
    def __init__(self, update_period: int = 1) -> None:

        self.frame_period_ns = update_period * 10**9
        self.last_change_pts = 0
        self.current_val = 0
        self.values = []

    def get_value(self, pts: int, new_value) -> int:
        self.values.append(new_value)

        if pts - self.last_change_pts >= self.frame_period_ns:
            values = sorted(self.values)
            self.current_val = values[len(values) // 2]
            self.values = []
            self.last_change_pts = pts

        return self.current_val
