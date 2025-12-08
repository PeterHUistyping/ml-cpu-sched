from src.sim.task.base_task import BaseTask
import matplotlib.pyplot as plt
from src.utils.plot_style import set_plot_style
import random


class TaskFactory:
    '''
        A factory class for creating BaseTask objects.

            Requires the automatic incrementing of task IDs over [1, 2, 3, ..., n_tasks],
            Random assignment of arrival times and sizes.
    '''
    def __init__(self, n_tasks: int):

        self.n_tasks = n_tasks
        self.tasks = []

        # fixed seed for reproducibility
        random.seed(42)


    def sort_tasks_by_arrival_time(self):
        '''
            Sort the tasks by their arrival time in ascending order.
        '''
        self.tasks.sort(key=lambda x: x.arrival_time)


    def create_tasks(self, SORT_BY_ARRIVAL_TIME=True) -> list[BaseTask]:
        '''
            Create n_tasks tasks with random arrival times and sizes.
        '''
        for i in range(1, self.n_tasks + 1):
            # create a random number 
            arrival_time = random.uniform(0, 10)  # e.g., arrival time between 0 and 100
            size = random.uniform(1, 8)  # e.g., size between 1 and 10
            task = BaseTask(task_id=i, arrival_time=arrival_time, size=size)
            # task.print_info()
            self.tasks.append(task)

        # sort tasks by arrival time
        if SORT_BY_ARRIVAL_TIME:
            self.sort_tasks_by_arrival_time()

        return self.tasks
    

    def visualize_tasks(self, output_dir='outputs/'):
        '''
            Plot the task duration in 1D plot, where x axis is time from start to end, and each task is represented as a rectangle starting with arrival_time and width=size.
        '''
        set_plot_style()
        arrival_times = [task.arrival_time for task in self.tasks]
        sizes = [task.size for task in self.tasks]
        plt.figure(figsize=(10, 6))
        for i, task in enumerate(self.tasks):
            color = plt.cm.tab20(task.task_id % 20)

            plt.barh(y=i, width=task.size, left=task.arrival_time, height=0.4, align='center', alpha=0.7, color=color) 
            plt.text(task.arrival_time + task.size / 2, i, f'Task {task.task_id}', va='center', ha='center', color='black')
        plt.xlabel('Time')
        plt.ylabel('Tasks')
        plt.title('Tasks Visualization')
        plt.yticks(range(len(self.tasks)), [f'Task {task.task_id}' for task in self.tasks])
        plt.tight_layout()
        plt.savefig(f'{output_dir}/task_factory_visualization.png')
        plt.close()


if __name__ == "__main__":
    factory = TaskFactory(n_tasks=10)
    factory.create_tasks(SORT_BY_ARRIVAL_TIME=False)
    factory.visualize_tasks()