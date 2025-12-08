import simpy
from typing import List

from src.sim.scheduler.base_scheduler import BaseScheduler
from src.sim.scheduler.round_robin_scheduler import RoundRobinScheduler
from src.sim.processor.base_processor import BaseProcessor
from src.sim.task.base_task import BaseTask
from src.sim.task.task_factory import TaskFactory
from src.utils.logging import *
logger = logging.getLogger("simulator")


# create ENUM for different scheduling strategies
class SchedulingStrategy:
    FCFS = "FCFS"
    ROUND_ROBIN = "RR"


# scheduling_strategy = SchedulingStrategy.FCFS
scheduling_strategy = SchedulingStrategy.ROUND_ROBIN
n_tasks = 10
n_processors = 1
t_simulation_end = 100


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


def write_results_html(output_dir="outputs/"):
    '''
        Write the visualization files into an HTML report.

        # Task Factory Visualization
        task_factory_visualization.png

        # Scheduler <scheduling_strategy> Processor Timelines
        processor_0_timeline_<scheduling_strategy>.png

        # append simulation.log contents
    '''
    with open(f"{output_dir}sim_report_{scheduling_strategy}.html", "w") as f:
        f.write("<html><head><title>Simulation Report</title></head><body>\n")
        f.write("<h1>Simulation Report</h1>\n")

        f.write("<h2>Task Factory Visualization</h2>\n")
        f.write('<img src="task_factory_visualization.png" alt="Task Factory Visualization"><br>\n')

        f.write(f"<h2>Scheduler {scheduling_strategy} Processor Timelines</h2>\n")
        for i in range(n_processors):
            f.write(f'<h3>Processor {i} Timeline</h3>\n')
            f.write(f'<img src="processor_{i}_timeline_{scheduling_strategy}.png" alt="Processor {i} Timeline"><br>\n')

        f.write("<h2>Simulation Log</h2>\n")
        f.write("<pre>\n")
        with open(f"{output_dir}simulation.log", "r") as log_file:
            f.write(log_file.read())
        f.write("</pre>\n") 

        f.write("</body></html>\n")


if __name__ == "__main__":
    
    env = simpy.Environment()
    logger.info("Simulator environment created.", extra={"env": env})

    task_factory = TaskFactory(n_tasks=n_tasks)
    tasks_list = task_factory.create_tasks()
    task_factory.visualize_tasks()
    logger.info(f"Created {len(tasks_list)} tasks.", extra={"tasks": tasks_list})

    processors = [BaseProcessor(env, processor_id=i, frequency=1.0) for i in range(n_processors)]

    # assign scheduler based on the selected strategy
    if scheduling_strategy == SchedulingStrategy.FCFS:
        scheduler = BaseScheduler(env, processors)
    elif scheduling_strategy == SchedulingStrategy.ROUND_ROBIN:
        scheduler = RoundRobinScheduler(env, processors, quantum=1.0)

    for processor in processors:
        processor.scheduler = scheduler  # link back the scheduler to the processor

    # adding the task gradually according to their arrival times
    env.process(task_arrival_generator(env, tasks_list, scheduler))

    env.run(until=t_simulation_end)
 
    print("Simulation completed.")

    for processor in processors:
        processor.visualization_plot(scheduling_strategy)

    write_results_html()

    


    