# main.py

import hydra
from pathlib import Path
from omegaconf import DictConfig
from utils.logging import create_logger
from utils.metrics_utils import evaluate_inverse_log_loss
from runners.bo_runner import OptimizerRunner


@hydra.main(version_base=None, config_path="../configs", config_name="config_cpu")
def main(args: DictConfig):
    n_trials = args.get("n_trials", 100)
    output_path = Path(args.get("output_path", "./outputs"))
    algo_name = args.get("algo_name", "RoundRobin")
    experiment_name = f"GP_BO_{algo_name}"

    db_filename = f"{experiment_name}_results.db"
    db_path = output_path / db_filename
    if not output_path.exists():
        output_path.mkdir(parents=True, exist_ok=True)

    storage_url = f"sqlite:///{db_path.resolve()}"

    logger = create_logger()
    logger.info("Starting Bayesian Optimization Experiment Setup...")
    logger.info(f"Database URL: {storage_url}")

    # Metric hyperparams
    metric_params = {
        "beta": args.get("loss_weight_beta", 1.0),
        "gamma": args.get("loss_weight_gamma", 1.0)
    }
    logger.info(f"Metric Parameters: {metric_params}")

    # Simulator constant
    simulation_params = {
        "n_tasks": args.get("n_tasks", 500),
        "t_simulation_end": args.get("t_simulation_end", 1000),
        "rate_lambda": args.get("rate_lambda", 1.0)

    }
    logger.info(f"Simulation Constants: {simulation_params}")

    try:
        runner = OptimizerRunner(
            metric_func=evaluate_inverse_log_loss,
            metric_params=metric_params,
            simulation_params=simulation_params,
            n_trials=n_trials,
            study_name=experiment_name,
            storage_path=storage_url
        )

        best_trial = runner.run()

        logger.info(
            f"\nOptimization complete. Best Trial ID: {best_trial.number}")
        logger.info(
            f"Optimal objective value achieved: {best_trial.value:.4f}")
        logger.info(f"Best Hyperparameters: {best_trial.params}")

    except Exception as e:
        logger.error(
            f"An error occurred during optimization: {e}", exc_info=True)


if __name__ == "__main__":
    main()
