from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Any
import simpy

from sim.task.base_task import BaseTask
from sim.processor.base_processor import BaseProcessor


class BaseScheduler(ABC):
    def __init__(self, env: simpy.Environment, resources: List[BaseProcessor]):
        self.env = env
        self.resources = resources
        self.queue = []  # Waiting queue

    def add_task(self, task: BaseTask):
        """External method to add a task to the system."""
        self.queue.append(task)
        self.schedule()  # Attempt to schedule whenever a new task arrives

    def on_resource_free(self, resource: BaseProcessor):
        """Triggered when a resource becomes free."""
        self.schedule()  # Resource is free, attempt to schedule

    @abstractmethod
    def select_next_task(self) -> Optional[BaseTask]:
        """[Core Strategy] Decide which task to select from the queue (FCFS, SJF, Priority...)."""
        pass

    @abstractmethod
    def select_resource(self, task: BaseTask) -> Optional[BaseProcessor]:
        """[Core Strategy] Decide which resource to assign the task to (Random, Least Loaded...)."""
        pass

    def schedule(self):
        """Generic scheduling loop."""
        # 1. Schedule only when there are tasks in the queue AND free resources
        available_resources = [r for r in self.resources if not r.busy]

        while self.queue and available_resources:
            # 2. Select task
            task = self.select_next_task()
            if not task:
                break

            # 3. Select resource
            resource = self.select_resource(task)
            if not resource:
                break

            # 4. Remove from queue and start execution
            if task in self.queue:
                self.queue.remove(task)

            # Mark this resource as booked
            available_resources.remove(resource)
            self.env.process(resource.process(task))
