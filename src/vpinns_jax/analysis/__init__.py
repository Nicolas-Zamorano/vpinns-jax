"""
Errors, training monitors, and plots of trained trial functions.
"""

from vpinns_jax.analysis.errors import compute_network_error, compute_solution_norm
from vpinns_jax.analysis.monitoring import TrainingRecord, create_training_monitor
from vpinns_jax.analysis.plotting import (
    plot_cell_errors,
    plot_fe_solution,
    plot_loss_terms,
    plot_network_solution,
    plot_robustness,
    plot_training_quantities,
)

__all__ = [
    "compute_network_error",
    "compute_solution_norm",
    "TrainingRecord",
    "create_training_monitor",
    "plot_training_quantities",
    "plot_loss_terms",
    "plot_robustness",
    "plot_network_solution",
    "plot_fe_solution",
    "plot_cell_errors",
]
