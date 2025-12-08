from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Any
import simpy

from src.sim.task.base_task import BaseTask
from src.sim.processor.base_processor import BaseProcessor
from src.sim.scheduler.base_scheduler import BaseScheduler
from src.utils.logging import *
logger = logging.getLogger("scheduler")


class RoundRobinScheduler(BaseScheduler):
    def __init__(self, env: simpy.Environment, resources: List[BaseProcessor], quantum: float = 1.0):
        super().__init__(env, resources)
        self.quantum = quantum  # Time quantum for round robin scheduling


    # @override
    def select_next_task(self) -> Optional[BaseTask]:
        """[Core Strategy] Decide which task to select from the queue via round robin."""
        if not self.queue:
            return None
        # Round Robin: simply pick the first task in the queue
        return self.queue[0]


    # @override
    def select_resource(self, task: BaseTask) -> Optional[BaseProcessor]:
        """[Core Strategy] Decide which resource to assign the task to, via the smallest ID."""
        available_resources = self.filter_available_resources()
        if not available_resources:
            return None
        # Select the resource with the smallest ID
        selected_resource = min(available_resources, key=lambda r: r.id)
        return selected_resource
    

    def post_schedule_hook(self, task, resource):
        # After scheduling, finish task after quantum or when done
        # re-add the task to the end of the queue if it's not finished
        self.env.process(resource.process(task, self.quantum))
        if task.remaining_size > 0:
            self.queue.append(task)


        