from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Any
import simpy

from sim.task.base_task import BaseTask


class BaseProcessor(ABC):
    def __init__(
        self,
        env: simpy.Environment,
        processor_id: int,
        capacity: float = 1.0
    ):
        self.env = env
        self.id = processor_id
        # Processing speed (e.g., CPU frequency or factory machine efficiency)
        self.capacity = capacity
        self.current_task: Optional[BaseTask] = None
        self.busy = False

    def process(self, task: BaseTask):
        """
        Generic processing logic: lock resource -> elapse time -> release resource
        """
        self.busy = True
        self.current_task = task
        task.start_time = self.env.now

        # Calculate how long it takes to process this task
        # Time = Size / Capacity
        processing_time = task.size / self.capacity

        # Simulate time passing (SimPy core)
        yield self.env.timeout(processing_time)

        # Task finished
        task.finish_time = self.env.now
        self.busy = False
        self.current_task = None

        # Hook: notify the scheduler that I am done
        self.on_task_finished(task)

    @abstractmethod
    def on_task_finished(self, task: BaseTask):
        """Callback function; concrete implementation injected by subclass or Scheduler"""
        pass
