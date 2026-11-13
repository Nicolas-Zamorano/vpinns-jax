"""
Errors, training monitors, error bounds, and plots of trained trial functions.
"""

from vpinns_jax.analysis.bounds import BoundRatios, ErrorBound, compute_bound_ratios
from vpinns_jax.analysis.errors import compute_network_error, compute_solution_norm
from vpinns_jax.analysis.monitoring import TrainingRecord, create_training_monitor
from vpinns_jax.analysis.plotting import (
    line_style,
    plot_cell_errors,
    plot_loss_terms,
    plot_network_solution,
    plot_robustness,
    plot_training_quantities,
    show_grid,
)
from vpinns_jax.analysis.quasi_interpolation import compute_quasi_interpolant_norm

__all__ = [
    "compute_network_error",
    "compute_solution_norm",
    "TrainingRecord",
    "create_training_monitor",
    "ErrorBound",
    "BoundRatios",
    "compute_bound_ratios",
    "compute_quasi_interpolant_norm",
    "plot_training_quantities",
    "plot_loss_terms",
    "plot_robustness",
    "plot_network_solution",
    "plot_cell_errors",
    "line_style",
    "show_grid",
]
