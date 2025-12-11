from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Any
import simpy

from sim.task.base_task import BaseTask
from sim.processor.base_processor import BaseProcessor
from sim.scheduler.base_scheduler import BaseScheduler


class PriorityScheduler(BaseScheduler):
    '''
        Non-preemptive Priority Scheduling Algorithm with quantum slice switching option.
    '''
    def __init__(self, env: simpy.Environment, resources: List[BaseProcessor], logging, quantum: float = 1.0, USE_QUANTUM: bool = False):
        super().__init__(env, resources, logging=logging)
        self.logger = logging.getLogger("Priority scheduler")
        self.quantum = quantum  # Time quantum [optional for priority scheduling]
        self.USE_QUANTUM = USE_QUANTUM  
        self.logger.info(f"Priority Scheduler initialized with quantum = {self.quantum}.", extra={"env": self.env})

    # @override

    def select_next_task(self) -> Optional[BaseTask]:
        """
            [Core Strategy] Decide which task to select from the queue via priority.
            The lower the priority value, the higher the priority.
        """
        if not self.queue:
            return None
        # Find the task with the highest priority (lowest priority value)
        self.queue.sort(key=lambda task: task.priority)
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

    # @override

    def post_schedule_hook(self, task: BaseTask, resource: BaseProcessor):
        """Hook for any post-scheduling actions."""
        if self.USE_QUANTUM:
            self.env.process(resource.process(task, quantum=self.quantum))
        else:
            self.env.process(resource.process(task))