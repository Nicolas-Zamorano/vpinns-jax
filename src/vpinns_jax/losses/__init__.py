"""Discrete residuals, Riesz loss, and boundary losses (implementation plan Component 4)."""

from vpinns_jax.losses.boundary_residual import boundary_residual_loss
from vpinns_jax.losses.euclidean import euclidean_loss
from vpinns_jax.losses.residual import (
    direct_trial_fe_residual,
    direct_trial_patch_residual,
    evaluate_trial_at_points,
    interpolated_trial_fe_residual,
    interpolated_trial_patch_residual,
)
from vpinns_jax.losses.riesz import (
    assemble_energy_gram_matrix,
    energy_bilinear_form,
    patch_self_energy,
    restrict_to_free_dofs,
    riesz_loss_global,
    riesz_loss_patch,
)

__all__ = [
    "evaluate_trial_at_points",
    "direct_trial_fe_residual",
    "interpolated_trial_fe_residual",
    "direct_trial_patch_residual",
    "interpolated_trial_patch_residual",
    "euclidean_loss",
    "energy_bilinear_form",
    "assemble_energy_gram_matrix",
    "restrict_to_free_dofs",
    "riesz_loss_global",
    "patch_self_energy",
    "riesz_loss_patch",
    "boundary_residual_loss",
]
