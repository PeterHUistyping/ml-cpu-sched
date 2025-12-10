# ML study for CPU scheduling

| Gaussian process | sensitivity analysis | Bayesian optimization |

[L48 MLPW project] A machine learning-based study for scheduling: Energy-performance-aware scheduling in heterogeneous multi-core systems.


## Installation

```shell
conda create -n ml-sched python=3.10 -y
conda activate ml-sched
pip install hydra-core
pip install botorch
pip install optuna
pip install optuna-integration
pip install optuna-dashboard
pip install simpy
pip install seaborn
```

## Run Simulation

To run the simulation, execute the following command with the desired global settings,

```shell
export PYTHONPATH="$(pwd)/src:$PYTHONPATH"

python src/sim/simulator.py
```

> The simulation results will be saved in the `outputs/` directory, including the `sim_report_<scheduling_strategy>.html` reports (visualizations and detailed logs).

## Run BO and Analysis

To run the bayesian optimization, use the script from `scripts/run_cpu.sh`.

The default search space is defined as follows:
```python
freq: float = range(0.8, 3.5)
scheduling_strategy: str = ["FCFS", "ROUND_ROBIN"]
quantum: float = range(0.5 - 5.0)

cost_func: Callable = evaluate_inverse_log_loss
beta: float = range(0.5, 5.0)
gamma: float = range(0.5, 5.0)
```

After the optimization is finished, use `optuna-dashboard` to check the results with corresponding analysis:
```shell
optuna-dashboard sqlite:///outputs/GP_BO_RoundRobin_results.db
```

## Code structure
- `src/sim/`
  - `simulator.py`: main simulation engine.
  - `task/`: define single task with its attributes and factory class to generate tasks.
  - `processor/`: define single processor with its attributes.
  - `scheduler/`: define various scheduling algorithms inheriting from the base scheduler, e.g. FCFS/FIFO, round-robin, priority-based, CFS, etc.