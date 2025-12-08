# main.py

import hydra
from pathlib import Path
from omegaconf import DictConfig
from utils.logging import create_logger
from utils.metrics_utils import evaluate_inverse_log_loss
from runners.bo_runner import OptimizerRunner


@hydra.main(version_base=None, config_path="../configs", config_name="config_rr")
def main(args: DictConfig):
    n_trials = args.get("n_trials", 50)
    output_path = Path(args.get("output_path"))
    algo_name = args.get("algo_name", "RoundRobin")
    experiment_name = f"GP_BO_{algo_name}"

    db_filename = f"../outputs/{experiment_name}_results.db"
    db_path = Path.cwd() / db_filename
    db_path.parent.mkdir(parents=True, exist_ok=True)
    storage_url = f"sqlite:///{db_path.resolve()}"

    logger = create_logger()
    logger.info("Starting Bayesian Optimization Experiment Setup...")
    logger.info(f"Database URL: {storage_url}")

    try:
        runner = OptimizerRunner(
            metric_func=evaluate_inverse_log_loss,
            n_trials=n_trials,
            study_name=experiment_name,
            storage_path=storage_url
        )

        best_trial = runner.run()

        logger.info(
            f"\nOptimization complete. Best Trial ID: {best_trial.number}")
        logger.info(
            f"Optimal objective value achieved: {best_trial.value:.4f}")

    except Exception as e:
        logger.error(
            f"An error occurred during optimization: {e}", exc_info=True)


if __name__ == "__main__":
    main()
