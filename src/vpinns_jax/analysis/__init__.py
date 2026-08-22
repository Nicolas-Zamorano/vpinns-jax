"""Convergence analysis and diagnostics (implementation plan Component 7)."""

from vpinns_jax.analysis.convergence import (
    ConvergenceRecord,
    plot_convergence_comparison,
    run_convergence_study,
)
from vpinns_jax.analysis.errors import (
    compute_energy_error,
    compute_h1_seminorm_error,
    compute_l2_error,
    compute_relative_l2_error,
    direct_trial_energy_error,
    direct_trial_h1_seminorm_error,
    direct_trial_l2_error,
    direct_trial_relative_l2_error,
    efficiency_index,
)

__all__ = [
    "compute_l2_error",
    "compute_relative_l2_error",
    "compute_h1_seminorm_error",
    "compute_energy_error",
    "direct_trial_l2_error",
    "direct_trial_relative_l2_error",
    "direct_trial_h1_seminorm_error",
    "direct_trial_energy_error",
    "efficiency_index",
    "ConvergenceRecord",
    "run_convergence_study",
    "plot_convergence_comparison",
]
