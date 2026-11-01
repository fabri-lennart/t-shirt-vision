class CrossingCounter:
    """Count each tracked object once after it crosses the configured line."""

    def __init__(self) -> None:
        self.seen_tracker_ids: set[int] = set()
        self.total = 0

    def record(self, tracker_id: int) -> bool:
        if tracker_id in self.seen_tracker_ids:
            return False
        self.seen_tracker_ids.add(tracker_id)
        self.total += 1
        return True
