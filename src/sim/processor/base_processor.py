from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Any
import simpy
import random
import matplotlib.pyplot as plt

from sim.task.base_task import BaseTask
from utils.plot_style import set_plot_style
from sim.processor.energy_consumption import calculate_energy_consumption


class BaseProcessor(ABC):
    '''
        Default processor class
    '''

    def __init__(
        self,
        env: simpy.Environment,
        processor_id: int,
        logging,
        frequency: float = 1.0,
    ):
        self.env = env
        self.id = processor_id
        self.logger = logging.getLogger(f"processor {self.id}")

        # Processing speed (e.g., CPU frequency or factory machine efficiency)
        self.frequency = frequency
        self.logger.info(f"Initial frequency = {self.frequency}.", extra={
                         "processor": self, "env": self.env})

        self.current_task: Optional[BaseTask] = None
        self.scheduler = None  # to be assigned after initialization.

        # if the processor is running a task
        self.busy = False

        # record of assignments for analysis: (start_time, finish_time, task)
        self.assignments = []  # type: List[tuple]

        # for idle energy calculation, if needed
        self.end_time = 0
        self.active_time = 0
        self.idle_time = 0
        self.idle_energy = 0

    def process(self, task: BaseTask, quantum: Optional[float] = None):
        """
        Generic processing logic: lock resource -> elapse time -> release resource
        """
        self.busy = True
        self.current_task = task

        start_time = self.env.now

        if task.start_time < 0:
            task.start_time = start_time

        # Calculate how long it takes to process this task
        self.logger.info(f"started processing Task {task.task_id} at time {start_time} with remaining size {task.remaining_size}.", extra={
                         "task": task, "env": self.env, "processor": self})

        processing_time = task.remaining_size / self.frequency
        if quantum is not None:
            processing_time = min(processing_time, quantum)
        reduced_size = min(processing_time * self.frequency,
                           task.remaining_size)
        task.remaining_size -= reduced_size

        # Calculate energy consumption
        energy_used = calculate_energy_consumption(processing_time, self.frequency)
        task.energy += energy_used

        # Simulate time passing (SimPy core)
        yield self.env.timeout(processing_time)

        self.logger.info(f"finished processing Task {task.task_id} at time {self.env.now}.", extra={
                         "task": task, "env": self.env, "processor": self})

        # Task finished
        finish_time = self.env.now
        self.busy = False
        self.current_task = None
        self.end_time = finish_time                     # update the end time
        self.active_time += (finish_time - start_time)
        
        # Hook: notify the scheduler that done
        self.on_task_finished(task)

        # Record the assignment
        self.assignments.append((start_time, finish_time, task))

    # @abstractmethod

    def on_task_finished(self, task: BaseTask):
        """Callback function; concrete implementation injected by subclass or Scheduler"""
        if self.scheduler:
            self.scheduler.on_resource_free(self, task)

    def visualization_plot(self, extra_args='', output_dir="outputs/"):
        """Helper method to visualize the processor's assignments."""
        set_plot_style()

        fig, ax = plt.subplots(figsize=(20, 2))
        for start, end, task in self.assignments:
            task_id = task.task_id
            color = task.color
            ax.broken_barh([(start, end - start)], (0, 5), facecolors=(color))
            # y position w.r.t the task id
            # depending on end-start, if too small, randomize y position
            if end - start < 0.5:
                y_pos = random.uniform(0.5, 4.5)
            else:
                y_pos = 2.5
            ax.text((start + end) / 2, y_pos,
                    f'{task_id}', ha='center', va='center', color='white')

        ax.set_ylim(0, 5)
        ax.set_xlim(0, max(end for _, end, _ in self.assignments) + 10)
        ax.set_xlabel('Time')
        ax.set_ylabel("Task ID", rotation=90)
        # remove all y ticks
        ax.set_yticks([])

        plt.setp(ax.get_yticklabels(), rotation=90,
                 ha='right', rotation_mode='anchor')
        ax.set_title(f'Processor {self.id} Assignment Timeline')
        plt.tight_layout()
        plt.savefig(
            f"{output_dir}processor_{self.id}_timeline_{extra_args}.png")
        plt.close()


    def get_idle_energy(self):
        # update idle time during runtime
        self.idle_time = self.end_time - self.active_time
        return calculate_energy_consumption(self.idle_time, self.frequency, is_idle=True)