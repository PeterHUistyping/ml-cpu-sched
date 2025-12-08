# runners/objective.py

import optuna
from typing import Callable

from sim.simulator import run_simulation


def objective(
    trial: optuna.Trial,
    metric_func: Callable
) -> float:

    # Get env inputs
    freq = trial.suggest_float("freq_ghz", 0.8, 3.5, log=True)
    env_type = trial.suggest_categorical(
        "scheduling_strategy", ["FCFS", "ROUND_ROBIN"])

    # Process with params
    args = {}
    if env_type == "ROUND_ROBIN":
        # 仅在 RR 被选中时采样 quantum
        args['quantum'] = trial.suggest_float("quantum_ms", 0.5, 5.0)

    results = run_simulation(env_type=env_type, freq=freq, **args)

    # Metric & Cost Calculation
    beta = trial.suggest_float("loss_weight_beta", 0.5, 5.0)
    gamma = trial.suggest_float("loss_weight_gamma", 0.5, 5.0)
    cost = metric_func(
        energy=results["avg_energy"],
        time=results["avg_duration"],
        beta=beta,
        gamma=gamma
    )

    return cost
