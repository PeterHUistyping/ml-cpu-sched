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
    algo_name = args.experiment.get("experiment_name", "")
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
        "beta": args.experiment.get("loss_weight_beta", 1.0),
        "gamma": args.experiment.get("loss_weight_gamma", 1.0)
    }
    logger.info(f"Metric Parameters: {metric_params}")

    # Simulator constant
    simulation_params = {
        "n_tasks": args.get("n_tasks", 500),
        "t_simulation_end": args.get("t_simulation_end", 1000),
        "rate_lambda": args.experiment.get("rate_lambda", 1.0)
    }
    logger.info(f"Simulation Constants: {simulation_params}")

    kernel_type = args.experiment.get("kernel_type", "matern_2.5")
    n_startup_trials = args.get("n_startup_trials", 5)
    objective_type = args.experiment.get("objective_type", "default")

    try:
        runner = OptimizerRunner(
            metric_func=evaluate_inverse_log_loss,
            metric_params=metric_params,
            simulation_params=simulation_params,
            n_trials=n_trials,
            study_name=experiment_name,
            storage_path=storage_url,
            kernel_type=kernel_type,
            n_startup_trials=n_startup_trials,
            objective_type=objective_type,
        )

        best_results = runner.run()

        if objective_type == "multi":
            logger.info("\n" + "="*50)
            logger.info(f"🎉 Multi-Objective Optimization Complete.")
            logger.info(
                f"Found {len(best_results)} solutions on the Pareto Front.")

            for i, trial in enumerate(best_results):
                logger.info(
                    f"\n--- Pareto Solution {i+1} (ID: {trial.number}) ---")
                # values 是一个列表 [Energy, Time]
                logger.info(f"Objectives [Energy, Time]: {trial.values}")
                logger.info(f"Params: {trial.params}")

            logger.info("="*50)

        else:
            best_trial = best_results

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
