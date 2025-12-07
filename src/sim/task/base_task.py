from dataclasses import dataclass
from typing import Optional, Any


@dataclass
class BaseTask:
    task_id: int
    arrival_time: float
    size: float
    # This is a generic concept:
    # - For CPU, size = number of cycles / instructions
    # - For Network, size = packet size (MB)
    # - For Factory, size = processing time

    # Record task lifecycle metrics
    start_time: Optional[float] = None
    finish_time: Optional[float] = None

    @property
    def duration(self):
        """Calculate actual duration (Wait + Service)."""
        if self.finish_time and self.arrival_time:
            return self.finish_time - self.arrival_time
        return 0.0
