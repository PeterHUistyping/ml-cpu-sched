# ML study for CPU scheduling

| Gaussian process | sensitivity analysis | Bayesian optimization |

[L48 MLPW project] A machine learning-based study for scheduling: Energy-performance-aware scheduling in heterogeneous multi-core systems.


### Installation

```shell
conda create -n ml-sched python=3.10 -y
conda activate ml-sched
pip install optuna
pip install seaborn
```

## Code structure
- `src/sim/`
  - `simulator.py`: main simulation engine.
  - `task/`: define single task with its attributes and factory class to generate tasks.
  - `processor/`: define single processor with its attributes.
  - `scheduler/`: define various scheduling algorithms inheriting from the base scheduler, e.g. FCFS/FIFO, round-robin, priority-based, CFS, etc.