import simpy
from typing import List
import numpy as np

from src.sim.scheduler.base_scheduler import BaseScheduler
from src.sim.scheduler.round_robin_scheduler import RoundRobinScheduler
from src.sim.processor.base_processor import BaseProcessor
from src.sim.task.base_task import BaseTask
from src.sim.task.task_factory import TaskFactory
from src.utils.logging import create_logger 


# create ENUM for different scheduling strategies
class SchedulingStrategy:
    FCFS = "FCFS"
    ROUND_ROBIN = "RR"


# scheduling_strategy = SchedulingStrategy.FCFS
scheduling_strategy = SchedulingStrategy.ROUND_ROBIN
n_tasks = 10
n_processors = 1
t_simulation_end = 40
quantum = 1.0  # time quantum for Round Robin

# determine extra args for filename
extra_args = ''

if scheduling_strategy == SchedulingStrategy.ROUND_ROBIN:
    extra_args = f"{scheduling_strategy}_{quantum}"
else:
    extra_args = f"{scheduling_strategy}"

# create logger
logging = create_logger(extra_args=extra_args)
logger = logging.getLogger("simulator")


def task_arrival_generator(env: simpy.Environment, tasks: List[BaseTask], scheduler: RoundRobinScheduler):
    """
        Generates task arrivals into the scheduler at their specified arrival times.
    """

    for task in tasks:
        # wait until the task's arrival time
        yield env.timeout(task.arrival_time - env.now)

        # done waiting, now add the task to the scheduler
        logger.info(f"Task {task.task_id} arrived at time {env.now}, with size {task.size}.", extra={"task": task, "env": env})
        scheduler.add_task(task)


def write_results_html(extra_args='', output_dir="outputs/", WRITE_ANALYSIS=True,  WRITE_LOG=True):
    '''
        Write the visualization files into an HTML report.

        # Task Factory Visualization
        task_factory_visualization.png

        # Scheduler <scheduling_strategy> Processor Timelines
        processor_0_timeline_<scheduling_strategy>.png

        # append simulation.log contents
    '''
    with open(f"{output_dir}sim_report_{extra_args}.html", "w") as f:
        f.write(f"<html><head><title>{scheduling_strategy} Sim.</title></head><body>\n")
        f.write("<h1>Simulation Report</h1>\n")

        f.write("<h2>Task Factory Visualization</h2>\n")
        f.write('<img src="task_factory_visualization.png" alt="Task Factory Visualization"><br>\n')

        f.write(f"<h2>Scheduler {scheduling_strategy} Processor Timelines</h2>\n")
        for i in range(n_processors):
            f.write(f'<h3>Processor {i} Timeline</h3>\n')
            f.write(f'<img src="processor_{i}_timeline_{scheduling_strategy}.png" alt="Processor {i} Timeline"><br>\n')

        if WRITE_ANALYSIS:
            f.write("<h2>Analysis Results</h2>\n")
            f.write("<pre>\n")
            with open(f"{output_dir}analysis_{extra_args}.txt", "r") as analysis_file:
                f.write(analysis_file.read())
            f.write("</pre>\n")

        if WRITE_LOG:
            f.write("<h2>Simulation Log</h2>\n")
            f.write("<pre>\n")
            with open(f"{output_dir}simulation_{extra_args}.log", "r") as log_file:
                f.write(log_file.read())
            f.write("</pre>\n") 

            f.write("</body></html>\n")


def write_analysis_file(output_dir="outputs/"):
    '''
        Write analysis results into a text file.
    '''
    duration_list = []
    with open(f"{output_dir}analysis_{extra_args}.txt", "w") as f:

        for task in tasks_list:
            f.write(f"Task {task.task_id}: Duration = {task.duration} (Finish Time={task.finish_time} - Arrival Time={task.arrival_time}) | Total Size = {task.size}, \n")

            duration_list.append(task.duration)

        duration_numpy = np.array(duration_list)
        avg_duration = np.mean(duration_numpy)
        var_duration = np.var(duration_numpy)

        f.write(f"\nAverage Task Duration: {avg_duration}, Variance: {var_duration}\n") 
   

if __name__ == "__main__":
    
    env = simpy.Environment()
    logger.info("Simulator environment created.", extra={"env": env})

    task_factory = TaskFactory(n_tasks=n_tasks)
    tasks_list = task_factory.create_tasks()
    task_factory.visualize_tasks()
    logger.info(f"Created {len(tasks_list)} tasks.", extra={"tasks": tasks_list})

    processors = [BaseProcessor(env, processor_id=i, logging=logging, frequency=1.0) for i in range(n_processors)]

    # assign scheduler based on the selected strategy
    if scheduling_strategy == SchedulingStrategy.FCFS:
        scheduler = BaseScheduler(env, processors, logging=logging)
    elif scheduling_strategy == SchedulingStrategy.ROUND_ROBIN:
        scheduler = RoundRobinScheduler(env, processors, logging=logging, quantum=quantum)

    for processor in processors:
        processor.scheduler = scheduler  # link back the scheduler to the processor

    # adding the task gradually according to their arrival times
    env.process(task_arrival_generator(env, tasks_list, scheduler))

    env.run(until=t_simulation_end)
 
    print("Simulation completed.")

    for processor in processors:
        processor.visualization_plot(scheduling_strategy)
    
    write_analysis_file()

    write_results_html(extra_args=extra_args)

