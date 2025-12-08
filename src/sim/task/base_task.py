from dataclasses import dataclass
from typing import Optional, Any


@dataclass
class BaseTask:
    '''
        This is a generic concept:
            For CPU, size = number of instructions
            For Network, size = packet size (MB)
            For Factory, size = processing time
    '''
    task_id: int
    arrival_time: float
    size: float

    # Record deadline or finish time if applicable, else None
    finish_time: Optional[float] = None


    @property
    def duration(self):
        """Calculate actual duration (Wait + Service)."""
        if self.finish_time and self.arrival_time:
            return self.finish_time - self.arrival_time
        return 0.0
    

    def print_info(self):
        print(f"Task ID: {self.task_id}")
        print(f"\t Size: {self.size}")
        print(f"\t Arrival Time: {self.arrival_time}, Finish Time: {self.finish_time}")


if __name__ == "__main__":
    # Example usage
    task = BaseTask(task_id=1, arrival_time=0.0, size=10.0, finish_time=5.0)
    task.print_info()