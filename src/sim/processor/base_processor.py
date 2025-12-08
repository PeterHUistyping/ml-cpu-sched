from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Any
import simpy
import matplotlib.pyplot as plt

from src.sim.task.base_task import BaseTask
from src.utils.logging import *
from src.utils.plot_style import set_plot_style 


class BaseProcessor(ABC):
    '''
        Default processor class
    '''
    def __init__(
        self,
        env: simpy.Environment,
        processor_id: int,
        frequency: float = 1.0,
        scheduler: Optional[Any] = None,
    ):
        self.env = env
        self.id = processor_id
        self.logger = logging.getLogger(f"processor {self.id}")

        # Processing speed (e.g., CPU frequency or factory machine efficiency)
        self.frequency = frequency
        self.current_task: Optional[BaseTask] = None

        self.scheduler = scheduler  # type: Optional[BaseScheduler]

        # if the processor is running a task
        self.busy = False

        # record of assignments for analysis
        # (start_time, finish_time, task)
        self.assignments = []  # type: List[tuple]


    def process(self, task: BaseTask, quantum: Optional[float] = None):
        """
        Generic processing logic: lock resource -> elapse time -> release resource
        """
        self.busy = True
        self.current_task = task

        start_time = self.env.now

        # Calculate how long it takes to process this task
        self.logger.info(f"started processing Task {task.task_id} at time {start_time}.", extra={"task": task, "env": self.env, "processor": self})

        processing_time = task.size / self.frequency
        if quantum is not None:
            processing_time = min(processing_time, quantum)
        reduced_size = min(processing_time * self.frequency, task.remaining_size)
        task.remaining_size -= reduced_size

        # Simulate time passing (SimPy core)
        yield self.env.timeout(processing_time)

        self.logger.info(f"finished processing Task {task.task_id} at time {self.env.now}.", extra={"task": task, "env": self.env, "processor": self})

        # Task finished
        finish_time = self.env.now
        self.busy = False
        self.current_task = None

        # Hook: notify the scheduler that done
        self.on_task_finished(task)

        # Record the assignment
        self.assignments.append((start_time, finish_time, task))


    # @abstractmethod
    def on_task_finished(self, task: BaseTask):
        """Callback function; concrete implementation injected by subclass or Scheduler"""
        if self.scheduler:
            self.scheduler.on_resource_free(self)
    

    def visualization_plot(self, scheduling_strategy='', output_dir = "outputs/"):
        """Helper method to visualize the processor's assignments."""
        set_plot_style()

        fig, ax = plt.subplots(figsize=(10, 2))
        for start, end, task in self.assignments:
            task_id = task.task_id
            color = task.color 
            ax.broken_barh([(start, end - start)], (0, 5), facecolors=(color))
            ax.text((start + end) / 2, 2.5, f'{task_id}', ha='center', va='center', color='white')

        ax.set_ylim(0, 5)
        ax.set_xlim(0, max(end for _, end, _ in self.assignments) + 10)
        ax.set_xlabel('Time')
        ax.set_ylabel("Task ID", rotation=90)
        # remove all y ticks
        ax.set_yticks([])

        plt.setp(ax.get_yticklabels(), rotation=90, ha='right', rotation_mode='anchor')
        ax.set_title(f'Processor {self.id} Assignment Timeline')
        plt.tight_layout()
        plt.savefig(f"{output_dir}processor_{self.id}_timeline_{scheduling_strategy}.png")
        plt.close()
       


