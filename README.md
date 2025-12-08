# ML study for CPU scheduling

| Gaussian process | sensitivity analysis | Bayesian optimization |

[L48 MLPW project] A machine learning-based study for scheduling: Energy-performance-aware scheduling in heterogeneous multi-core systems.


## Installation

```shell
conda create -n ml-sched python=3.10 -y
conda activate ml-sched
pip install optuna
pip install seaborn
```

## Run 

To run the simulation, execute the following command with the desired global settings,

```shell
export PYTHONPATH="$(pwd):$PYTHONPATH"

python src/sim/simulator.py
```

> The simulation results will be saved in the `outputs/` directory, including the `sim_report_<scheduling_strategy>.html` reports (visualizations and detailed logs).

## Code structure
- `src/sim/`
  - `simulator.py`: main simulation engine.
  - `task/`: define single task with its attributes and factory class to generate tasks.
  - `processor/`: define single processor with its attributes.
  - `scheduler/`: define various scheduling algorithms inheriting from the base scheduler, e.g. FCFS/FIFO, round-robin, priority-based, CFS, etc.