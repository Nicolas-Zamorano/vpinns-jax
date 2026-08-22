"""A posteriori error estimator and marking rules (implementation plan Component 5)."""

from vpinns_jax.estimators.aposteriori import (
    compute_residual_error_estimator,
    direct_trial_patch_residual_indicators,
    direct_trial_strong_residual,
    global_patch_estimator,
    interpolated_trial_patch_residual_indicators,
    patch_indicators_to_cell_indicators,
    patch_loss_indicators,
)
from vpinns_jax.estimators.marking import cap_marked_fraction, mark_by_dorfler_bulk_criterion

__all__ = [
    "compute_residual_error_estimator",
    "patch_loss_indicators",
    "direct_trial_strong_residual",
    "direct_trial_patch_residual_indicators",
    "interpolated_trial_patch_residual_indicators",
    "global_patch_estimator",
    "patch_indicators_to_cell_indicators",
    "mark_by_dorfler_bulk_criterion",
    "cap_marked_fraction",
]
