# runners/objective.py

import optuna
from typing import Callable, Dict

from sim.simulator import run_simulation


PROCESSOR_TYPES = ["little", "medium", "big"]


def objective(
    trial: optuna.Trial,
    metric_func: Callable,
    metric_params: Dict[str, float] = None
) -> float:
    """
    Optuna Objective Function for System Optimization.

    Args:
        trial: Optuna trial object.
        metric_func: The evaluation function (e.g., evaluate_inverse_log_loss).
        metric_params: Fixed parameters for the metric (e.g., {'beta': 1.0, 'gamma': 2.0}).
                       These should be FIXED, not sampled, to ensure a stable optimization goal.
    """
    if metric_params is None:
        metric_params = {}

    # System Architecture Search)
    processor_freqs = []
    processor_counts = []

    for p_type in PROCESSOR_TYPES:
        # Frequency
        if p_type == "little":
            freq = trial.suggest_float(
                f"freq_{p_type}_ghz", 0.5, 1.5, step=0.1)
        elif p_type == "medium":
            freq = trial.suggest_float(
                f"freq_{p_type}_ghz", 1.0, 2.5, step=0.1)
        else:  # big
            freq = trial.suggest_float(
                f"freq_{p_type}_ghz", 1.5, 3.5, step=0.1)

        # Core Counts
        count = trial.suggest_int(f"count_{p_type}", 0, 4)

        processor_freqs.append(freq)
        processor_counts.append(count)

    if sum(processor_counts) == 0:
        processor_counts[0] = 1

    # Scheduling Strategy Search)
    env_type = trial.suggest_categorical(
        "scheduling_strategy", ["FCFS", "ROUND_ROBIN", "PRIORITY"])

    sim_args = {}

    if env_type in ["ROUND_ROBIN", "PRIORITY"]:
        sim_args['quantum'] = trial.suggest_float("quantum_ms", 2.0, 20.0)

    # Run Simulation)
    try:
        results = run_simulation(
            env_type=env_type,
            processor_freqs=processor_freqs,
            processor_counts=processor_counts,
            **sim_args
        )
    except Exception as e:
        print(f"Simulation failed: {e}")
        return float('inf')

    # Metric Calculation
    energy = results["avg_energy"]
    time = results["avg_duration"]

    cost = metric_func(energy=energy, time=time, **metric_params)

    return cost
