# runners/bo_runner.py

import optuna
import logging
from typing import Callable, Dict, Any, Optional
from optuna_integration.botorch import BoTorchSampler

from .objective import objective
from utils.metrics_utils import evaluate_inverse_log_loss

logger = logging.getLogger(__file__)


class OptimizerRunner:
    def __init__(
        self,
        metric_func: Callable = evaluate_inverse_log_loss,
        n_trials: int = 50,
        study_name: str = "GP_BO_Scheduler_Tuning",
        storage_path: Optional[str] = None
    ):
        """
        Setup Runner for BO experiment. 

        Args:
            metric_func (Callable): Metric func to calculate cost. 
            n_trials (int): Times of BO.
            study_name (str): Optuna Study 的名称。
            storage_path (str): Database path to store the results. 
        """
        self.metric_func = metric_func
        self.n_trials = n_trials
        self.study_name = study_name
        self.storage_path = storage_path

        self.sampler = BoTorchSampler()
        logger.info("Initialize BoTorchSampler...")

    def run(self) -> optuna.Trial:
        """
        Create Optuna Study and begin optimization pipeline. 
        """
        # Initialize optuna study -> minimize the cost.
        study = optuna.create_study(
            direction="minimize",
            sampler=self.sampler,
            study_name=self.study_name,
            storage=self.storage_path
        )

        # Run BO.
        logger.info(f"🚀 Begin '{self.n_trials}' times BO optimization... ")

        study.optimize(
            lambda trial: objective(trial, self.metric_func),
            n_trials=self.n_trials,
            show_progress_bar=True
        )

        best_params_str = ""
        for key, value in study.best_params.items():
            if isinstance(value, (int, float)):
                formatted_value = f"{value:.4f}"
            else:
                formatted_value = str(value)

            best_params_str += f"\n- {key}: {formatted_value}"

        logger.info("\n" + "="*50)
        logger.info(f"🎉 BO complete. Best value: {study.best_value:.4f}")
        logger.info("Optimal Config Params:")
        logger.info(best_params_str)
        logger.info("="*50)

        return study.best_trial
