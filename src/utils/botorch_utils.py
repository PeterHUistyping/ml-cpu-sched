# utils/botorch_utils.py

import torch
import logging
from typing import Optional
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import MaternKernel, RBFKernel, ScaleKernel
from botorch.acquisition import LogExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.exceptions.errors import ModelFittingError
from botorch.models.transforms import Standardize, Normalize

logger = logging.getLogger(__name__)


def get_covar_module(kernel_type: str, ard_num_dims: int):
    """
    Constructs a GPyTorch Kernel based on the provided name.

    Args:
        kernel_type: 'rbf', 'matern_1.5', 'matern_2.5'
        ard_num_dims: Number of input dimensions (used for Automatic Relevance Determination - ARD)
    """
    kernel_type = kernel_type.lower()

    if kernel_type == "rbf":
        # Radial Basis Function (Squared Exponential)
        base_kernel = RBFKernel(ard_num_dims=ard_num_dims)
    elif kernel_type == "matern_1.5":
        base_kernel = MaternKernel(nu=1.5, ard_num_dims=ard_num_dims)
    elif kernel_type == "matern_2.5":
        base_kernel = MaternKernel(nu=2.5, ard_num_dims=ard_num_dims)
    else:
        logger.warning(
            f"Unknown kernel '{kernel_type}', falling back to Matern 2.5")
        base_kernel = MaternKernel(nu=2.5, ard_num_dims=ard_num_dims)

    # ScaleKernel is used for adaptive output amplitude scaling
    return ScaleKernel(base_kernel)


def custom_candidate_func(
    train_x: torch.Tensor,
    train_obj: torch.Tensor,
    train_con: Optional[torch.Tensor],
    bounds: torch.Tensor,
    pending_x: Optional[torch.Tensor] = None,
    kernel_type: str = "matern_2.5"
) -> torch.Tensor:
    """
    Custom candidate generation function to be called by Optuna BoTorchSampler.
    It is responsible for building the GP model, fitting hyperparameters, and optimizing the acquisition function.
    """
    train_x = train_x.to(dtype=torch.float64)
    train_obj = train_obj.to(dtype=torch.float64)
    bounds = bounds.to(dtype=torch.float64)

    input_dim = train_x.shape[-1]

    # Build GP model with custom Kernel
    covar_module = get_covar_module(kernel_type, ard_num_dims=input_dim)

    model = SingleTaskGP(
        train_X=train_x,
        train_Y=train_obj,
        covar_module=covar_module,
        input_transform=Normalize(d=input_dim, bounds=bounds),
        outcome_transform=Standardize(m=train_obj.shape[-1])
    )

    # Fit the model (Compute hyperparameters)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    try:
        fit_gpytorch_mll(mll)
    except ModelFittingError as e:
        logger.warning(
            f"GP fitting failed: {e}. Proceeding with initialized hyperparameters.")
        pass

    # 4. Define Acquisition Function (Using LogExpectedImprovement for better numerical stability)
    # Note: train_obj passed by Optuna is already processed (maximization target), so we use it directly.
    acq_f = LogExpectedImprovement(
        model=model,
        best_f=train_obj.max(),
    )

    # 5. Optimize the acquisition function to find the next sampling point
    candidates, _ = optimize_acqf(
        acq_function=acq_f,
        bounds=bounds,
        q=1,  # Suggest 1 point at a time
        num_restarts=10,
        raw_samples=512,  # Number of initial samples; increasing improves global search capability
    )

    return candidates
