from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Any
import simpy

from src.sim.task.base_task import BaseTask
from src.sim.processor.base_processor import BaseProcessor


class BaseScheduler(ABC):
    '''
        Default first come first serve scheduler implementation, without interruptions.
    '''
    def __init__(self, env: simpy.Environment, resources: List[BaseProcessor], logging):
        self.env = env
        self.resources = resources
        self.queue = []  # type: List[BaseTask]
        self.logger = logging.getLogger("scheduler")


    def add_task(self, task: BaseTask):
        """External method to add a task to the system."""
        self.logger.info(f"Task {task.task_id} added to scheduler queue at time {self.env.now}.", extra={"task": task, "env": self.env})
        self.queue.append(task)
        self.schedule()  # Attempt to schedule whenever a new task arrives


    def on_resource_free(self, resource: BaseProcessor, task):
        """Triggered when a resource becomes free."""

        # After scheduling, finish task after quantum/interrupt or when done
        # re-add the task to the end of the queue if it's not finished
        if task.remaining_size > 0:
            self.queue.append(task)
            self.logger.info(f"Task {task.task_id} quantum expired at time {self.env.now}, re-adding to queue with remaining size {task.remaining_size}.", extra={"task": task, "env": self.env})
        else:
            task.finish_time = self.env.now
            self.logger.info(f"Task {task.task_id} completed at time {self.env.now}.", extra={"task": task, "env": self.env})

        self.schedule()  # Resource is free, attempt to schedule


    # @abstractmethod
    def select_next_task(self) -> Optional[BaseTask]:
        """[Core Strategy] Decide which task to select from the queue (FCFS, SJF, Priority...)."""
        # select the first one as default
        if not self.queue:
            return None
        return self.queue[0]
        

    def filter_available_resources(self) -> List[BaseProcessor]:
        """Helper method to filter and return available resources."""
        available_resources = [r for r in self.resources if not r.busy]
        if not available_resources:
            return None
        else:
            return available_resources


    # @abstractmethod
    def select_resource(self, task: BaseTask) -> Optional[BaseProcessor]:
        """[Core Strategy] Decide which resource to assign the task to (Random, Least Loaded...)."""
        available_resources = self.filter_available_resources()
        if not available_resources:
            return None
        return available_resources[0]


    def post_schedule_hook(self, task: BaseTask, resource: BaseProcessor):
        """Hook for any post-scheduling actions."""
        self.env.process(resource.process(task))


    def schedule(self):
        """Generic scheduling loop."""
        # 1. Schedule only when there are tasks in the queue AND free resources
        available_resources = self.filter_available_resources()

        while self.queue and available_resources:
            # 2. Select task
            task = self.select_next_task()
            if not task:
                break

            # 3. Select resource
            resource = self.select_resource(task)
            if resource is None:
                break

            # 4. Remove from queue and start execution
            if task in self.queue:
                self.queue.remove(task)

            # Mark this resource as booked
            available_resources.remove(resource)
            self.logger.info(f"Scheduling Task {task.task_id} on Resource {resource.id} at time {self.env.now}.", extra={"task": task, "resource": resource, "env": self.env})
            self.post_schedule_hook(task, resource)
