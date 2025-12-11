import simpy
from typing import List
import numpy as np

# update a little bit with import path...
from sim.scheduler.base_scheduler import BaseScheduler
from sim.scheduler.round_robin_scheduler import RoundRobinScheduler
from sim.scheduler.priority_scheduler import PriorityScheduler
from sim.processor.base_processor import BaseProcessor
from sim.task.base_task import BaseTask
from sim.task.task_factory import TaskFactory
from utils.logging import create_logger

from typing import Dict


# create ENUM for different scheduling strategies
class SchedulingStrategy:
    FCFS = "FCFS"
    ROUND_ROBIN = "RR"
    PRIORITY = "PRIORITY"


# scheduling_strategy = SchedulingStrategy.FCFS
scheduling_strategy = SchedulingStrategy.ROUND_ROBIN
# scheduling_strategy = SchedulingStrategy.PRIORITY
n_tasks = 10
n_processor_types = 3                             # little, medium, big 
n_processors_per_type = [2] * n_processor_types   # number of processors for each type
# or use [2, 2, 2] 
n_processors = sum(n_processors_per_type)
t_simulation_end = 100
# Processor frequency: here we assume the frequency is fixed along the task execution for simplicity.
frequencies = [0.5, 1.0, 2.0]  # processor frequency (GHz) for little, medium, big
quantum = 1.0       # time quantum for Round Robin (ms)

# determine extra args for filename
extra_args = f"{scheduling_strategy}_freq={frequencies}"

if scheduling_strategy == SchedulingStrategy.ROUND_ROBIN:
    extra_args += f"_quantum={quantum}"

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
        logger.info(f"Task {task.task_id} arrived at time {env.now}, with size {task.size}.", extra={
                    "task": task, "env": env})
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
        f.write(f"<html><head><title>{extra_args} Sim.</title></head><body>\n")
        f.write("<h1>Simulation Report</h1>\n")

        f.write("<h2>Task Factory Visualization</h2>\n")
        f.write(
            '<img src="task_factory_visualization.png" alt="Task Factory Visualization"><br>\n')

        f.write(
            f"<h2>Scheduler {scheduling_strategy} Processors Timelines</h2>\n")
        for i in range(n_processors):
            f.write(f'<h3>Processor {i} Timeline ({processors[i].frequency} GHz)</h3>\n')
            # write processor[i].end_time, processors[i].active_time, processors[i].get_idle_energy()
            f.write(f"<p>Idle Energy Consumption: {processors[i].get_idle_energy()} J, [End Time: {processors[i].end_time} s, Active Time: {processors[i].active_time} s, Idle Time: {processors[i].idle_time} s].</p>\n")
            f.write(
                f'<img src="processor_{i}_timeline_{extra_args}.png" alt="Processor {i} Timeline"><br>\n')

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


def get_mean_and_variance(data: List[float]) -> Dict[str, float]:
    '''
        parameters:
            data: list of numbers [float]

        Calculate mean and variance of a list of numbers.
    '''
    data_numpy = np.array(data)
    mean = np.mean(data_numpy)
    variance = np.var(data_numpy)
    return mean, variance


def weighted_time_by_priority(time, priority):
    '''
        Calculate weighted time by priority.
        Higher priority (lower value) gets higher weight, e.g. {0, 1, 2, ...} -> {1, 0.5, 0.33, ...}
    '''
    weight = 1 / ((priority + 1)**2)  # avoid division by zero
    return time * weight


def write_analysis_file(output_dir="outputs/", WRITE_SINGLE_TASK_ANALYSIS=True):
    '''
        Write analysis results into a text file.
    '''
    # turn around time
    duration_list = []
    weight_duration_list = []
    response_time_list = []
    energy_list = []
    idle_energy_list = []
    active_energy_list = []
    with open(f"{output_dir}analysis_{extra_args}.txt", "w") as f:

        f.write(f"Simulation Analysis Results for {extra_args}\n")

        for task in tasks_list:

            response_time = task.start_time - task.arrival_time
            weighted_duration = weighted_time_by_priority(task.duration, task.priority)
    
            if WRITE_SINGLE_TASK_ANALYSIS:
                f.write(
                    f"[Task {task.task_id}] turn around time = {task.duration} (finish={task.finish_time} - arrival={task.arrival_time}) \n \t total task size = {task.size}, \n")

                f.write(
                    f"\t response time = {response_time} (start={task.start_time} - arrival={task.arrival_time})\n")
                
                f.write(
                    f"\t weighted turn around time by priority (priority={task.priority}) = {weighted_duration}\n")

                f.write(f"\t energy consumption = {task.energy} J\n")

            duration_list.append(task.duration)
            weight_duration_list.append(weighted_duration)
            response_time_list.append(response_time)
            active_energy_list.append(task.energy)
        for processor in processors:
            idle_energy = processor.get_idle_energy()
            idle_energy_list.append(idle_energy)

        avg_duration, var_duration = get_mean_and_variance(duration_list)

        f.write(
            f"\n[turn around time] Average : {avg_duration}, Variance: {var_duration}\n")

        avg_response_time, var_response_time = get_mean_and_variance(response_time_list)
        f.write(
            f"[response time] Average : {avg_response_time}, Variance: {var_response_time}\n")
        
        avg_weighted_duration, var_weighted_duration = get_mean_and_variance(weight_duration_list)
        f.write(
            f"[weighted turn around time by priority] Average : {avg_weighted_duration}, Variance: {var_weighted_duration}\n")
        
        energy_list = [a + b for a, b in zip(active_energy_list, idle_energy_list)]
        avg_energy, var_energy = get_mean_and_variance(energy_list)
        f.write(
            f"[energy consumption] Average : {avg_energy} J, Variance: {var_energy} J\n")
        
        # also write active and idle energy separately
        avg_active_energy, var_active_energy = get_mean_and_variance(active_energy_list)
        f.write(
            f"\t[active energy consumption] Average : {avg_active_energy} J, Variance: {var_active_energy} J\n")
        
        avg_idle_energy, var_idle_energy = get_mean_and_variance(idle_energy_list)
        f.write(
            f"\t[idle energy consumption] Average : {avg_idle_energy} J, Variance: {var_idle_energy} J\n")


def run_simulation(env_type: str, freq: float, **args) -> Dict[str, float]:
    """
    Main interface for running scheduling task simulation.

    Args:
        env_type (str): Type of the simulation environment ("FCFS" or "ROUND_ROBIN"). 
        freq (float): Frequency of CPU processor (GHz).
        **args: Other simulation arguments, including 'quantum' for ROUND_ROBIN.
    Return:
        Dict[str, float]: Performance results {"avg_energy", "avg_duration"}.
    """

    VALID_ENV = {
        "FCFS": SchedulingStrategy.FCFS,
        "ROUND_ROBIN": SchedulingStrategy.ROUND_ROBIN
    }

    assert env_type in VALID_ENV.keys(
    ), f"Environment '{env_type}' is not supported! "

    # initialize params
    t_simulation_end = args.get('t_simulation_end', 100)
    n_tasks = args.get('n_tasks', 10)
    n_processors = args.get('n_processors', 1)

    env = simpy.Environment()
    task_factory = TaskFactory(n_tasks=n_tasks)
    tasks_list = task_factory.create_tasks()

    # initialize processor
    processors = [
        BaseProcessor(env, processor_id=i, logging=logging, frequency=freq)
        for i in range(n_processors)
    ]

    # initialize scheduler
    if env_type == "FCFS":
        scheduler = BaseScheduler(env, processors, logging=logging)

    elif env_type == "ROUND_ROBIN":
        quantum = args.get('quantum')
        if quantum is None:
            raise ValueError(
                "Environment ROUND_ROBIN requires the 'quantum' argument in **args.")

        scheduler = RoundRobinScheduler(
            env, processors, logging=logging, quantum=quantum)

    # run simulation
    for processor in processors:
        processor.scheduler = scheduler
    env.process(task_arrival_generator(env, tasks_list, scheduler))
    env.run(until=t_simulation_end)

    duration_list = []
    weighted_duration_list = []
    energy_list = []
    active_energy_list = []
    idle_energy_list = []

    for task in tasks_list:
        duration_list.append(task.duration)

        # weighted turn around time by priority
        weighted_duration = weighted_time_by_priority(task.duration, task.priority)
        weighted_duration_list.append(weighted_duration)

        # energy consumption when core is active
        active_energy_list.append(task.energy)
    
    for processor in processors:
        # energy consumption when core is idle
        idle_energy = processor.get_idle_energy()
        idle_energy_list.append(idle_energy)

    # total energy consumption
    energy_list = [a + b for a, b in zip(active_energy_list, idle_energy_list)]
    # Return the metric for Optuna Objective func
    return {
        "avg_energy": np.mean(energy_list),
        "avg_duration": np.mean(duration_list)
    }


if __name__ == "__main__":

    env = simpy.Environment()
    logger.info("Simulator environment created.", extra={"env": env})

    task_factory = TaskFactory(n_tasks=n_tasks)
    tasks_list = task_factory.create_tasks()
    task_factory.visualize_tasks()
    logger.info(f"Created {len(tasks_list)} tasks.",
                extra={"tasks": tasks_list})


    processors = []
    processor_id = 0
    for p_type in range(n_processor_types):
        for _ in range(n_processors_per_type[p_type]):
            processors.append(BaseProcessor(
                env, processor_id=processor_id, logging=logging, frequency=frequencies[p_type]))
            processor_id += 1

    # assign scheduler based on the selected strategy
    if scheduling_strategy == SchedulingStrategy.FCFS:
        scheduler = BaseScheduler(env, processors, logging=logging)
    elif scheduling_strategy == SchedulingStrategy.ROUND_ROBIN:
        scheduler = RoundRobinScheduler(
            env, processors, logging=logging, quantum=quantum)
    elif scheduling_strategy == SchedulingStrategy.PRIORITY:
        scheduler = PriorityScheduler(
            env, processors, logging=logging, quantum=quantum, USE_QUANTUM=True)

    for processor in processors:
        processor.scheduler = scheduler  # link back the scheduler to the processor

    # adding the task gradually according to their arrival times
    env.process(task_arrival_generator(env, tasks_list, scheduler))

    env.run(until=t_simulation_end)

    print("Simulation completed.")

    for processor in processors:
        processor.visualization_plot(extra_args=extra_args)

    write_analysis_file()

    write_results_html(extra_args=extra_args)
