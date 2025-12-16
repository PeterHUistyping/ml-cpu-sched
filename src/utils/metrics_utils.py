import math


def evaluate_edp(energy: float, time: float, power: int = 1) -> float:
    """
    1. Energy-Delay Product (EDP)
    Classic academic metric.

    Formula: Cost = Energy * (Time ^ power)

    Args:
        power (int): Defaults to 1 (standard EDP).
                     If you prioritize performance more, set to 2 (i.e., ED^2P).
    """
    if time <= 0 or energy <= 0:
        return 0.0  # Penalize invalid states

    return energy * (time ** power)


def evaluate_weighted_sum(
    energy: float,
    time: float,
    alpha: float = 0.5,
    limits: dict = None
) -> float:
    """
    2. Weighted Sum
    Must normalize first; otherwise, Alpha will be ineffective due to the
    different orders of magnitude between Energy and Time.

    Formula: Cost = alpha * Norm(Time) + (1 - alpha) * Norm(Energy)

    Args:
        limits (dict): Contains 'min_e', 'max_e', 'min_t', 'max_t' for normalization.
                       If None, assumes input is already normalized, though this is usually risky.
    """
    if limits:
        # Min-Max Normalization: (x - min) / (max - min)
        # Add 1e-9 to prevent division by zero
        norm_e = (energy - limits['min_e']) / \
            (limits['max_e'] - limits['min_e'] + 1e-9)
        norm_t = (time - limits['min_t']) / \
            (limits['max_t'] - limits['min_t'] + 1e-9)
    else:
        # Warning: If no limits are provided, manual scaling is suggested (e.g., divide Energy by 1000).
        # Calculating with raw values for now (high risk).
        norm_e = energy
        norm_t = time

    # Larger Alpha values prioritize time (performance); smaller Alpha values prioritize energy.
    return (alpha * norm_t) + ((1 - alpha) * norm_e)


def evaluate_inverse_log_loss(
    energy: float,
    time: float,
    beta: float = 1.0,
    gamma: float = 1.0
) -> float:
    """
    3. Log-Space Inverse Utility
    Converts 'maximize score' to 'minimize log loss', offering the best numerical stability.

    Original Score = 1 / (E^beta * T^gamma)  (Higher is better)
    Log Loss       = beta * log(E) + gamma * log(T) (Lower is better)

    Args:
        beta:  Penalty weight for energy
        gamma: Penalty weight for time
    """
    if energy <= 0 or time <= 0:
        return 0.0

    # Using log addition instead of multiplication smoothes magnitude differences; ideal for optimizers.
    return beta * math.log(energy) + gamma * math.log(time)
