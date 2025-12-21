# runners/bo_runner.py

import optuna
import logging
from functools import partial
from typing import Callable, Dict, Any, Optional
from optuna_integration.botorch import BoTorchSampler

from .objective import objective
from utils.metrics_utils import evaluate_inverse_log_loss
from utils.botorch_utils import custom_candidate_func

logger = logging.getLogger(__file__)


class OptimizerRunner:
    def __init__(
        self,
        metric_func: Callable = evaluate_inverse_log_loss,
        metric_params: Dict[str, float] = None,
        simulation_params: Dict[str, Any] = None,
        n_trials: int = 50,
        study_name: str = "GP_BO_Scheduler_Tuning",
        storage_path: Optional[str] = None,
        kernel_type: str = "matern_2.5",
    ):
        """
        Setup Runner for BO experiment. 

        Args:
            metric_func (Callable): Metric func to calculate cost.
            metric_params (Dict): Fixed parameters for the metric function (e.g. beta=1.0).
            simulation_params (Dict): Fixed parameters for the simulator (e.g. n_tasks=10).
            n_trials (int): Times of BO.
            study_name (str): Name of Optuna Study.
            storage_path (str): Database path to store the results.
        """
        self.metric_func = metric_func
        self.metric_params = metric_params if metric_params is not None else {}
        self.simulation_params = simulation_params if simulation_params is not None else {}

        self.n_trials = n_trials
        self.study_name = study_name
        self.storage_path = storage_path
        self.kernel_type = kernel_type

        self._setup_sampler()
        logger.info("Initialize BoTorchSampler...")

    def _setup_sampler(self):
        """
        Configures the BoTorchSampler with the specific kernel logic.
        """
        logger.info(
            f"Initialize BoTorchSampler with kernel: {self.kernel_type}...")

        candidate_func_with_kernel = partial(
            custom_candidate_func,
            kernel_type=self.kernel_type
        )

        self.sampler = BoTorchSampler(
            candidate_func=candidate_func_with_kernel,
            n_startup_trials=5
        )

    def run(self) -> optuna.Trial:
        """
        Create Optuna Study and begin optimization pipeline. 
        """
        # Initialize optuna study -> minimize the cost.
        study = optuna.create_study(
            direction="minimize",
            sampler=self.sampler,
            study_name=self.study_name,
            storage=self.storage_path,
            load_if_exists=True
        )

        # Run BO.
        logger.info(f"🚀 Begin '{self.n_trials}' times BO optimization... ")

        target_func = partial(
            objective,
            metric_func=self.metric_func,
            metric_params=self.metric_params,
            **self.simulation_params
        )

        study.optimize(
            target_func,
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
