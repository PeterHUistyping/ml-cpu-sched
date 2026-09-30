# Machine Learning for Energy-Performance-aware Scheduling


A machine learning-based study for scheduling heterogeneous multi-core systems.

Zheyuan (Peter) Hu $^\dagger$, Yifei Shi $^\dagger$.

$^\dagger$ denotes equal contribution. Accepted by NeurIPS'26 MLForSys Workshop (Poster).

| Gaussian process | sensitivity analysis | Bayesian optimization |

- [Cambridge ACS L48 MLPW](https://www.cl.cam.ac.uk/teaching/2526/L48/) project, [course page](https://mlatcl.github.io/mlphysical/), [2526](https://carlhenrik.com/l48-mlpw/).
- We sincerely appreciate [Professor Carl Henrik Ek](https://carlhenrik.com/) for organizing this exciting module and providing consistent feedback regarding this project during the proposal phase.

If you find this project useful, please consider citing:

```
@misc{HuShi2026mlcpusched,
      title={Machine Learning for Energy-Performance-aware Scheduling}, 
      author={Zheyuan Hu and Yifei Shi},
      year={2026},
      eprint={2601.23134},
      archivePrefix={arXiv},
      primaryClass={cs.AR},
      url={https://arxiv.org/abs/2601.23134}, 
}
```

## Installation

Conda environment setup:
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

Pip environment setup:
```shell
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
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
optuna-dashboard sqlite:///results_db/GP_BO_optimize_metric_balance_results.db
```

## Code structure
- `src/sim/`
  - `simulator.py`: main simulation engine.
  - `task/`: define single task with its attributes and factory class to generate tasks.
  - `processor/`: define single processor with its attributes.
  - `scheduler/`: define various scheduling algorithms inheriting from the base scheduler, e.g. FCFS/FIFO, round-robin, priority-based, CFS, etc.
