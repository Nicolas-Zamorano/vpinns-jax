"""
Residual vectors and the losses built from them.
"""

from vpinns_jax.losses.boundary_residual import boundary_residual_loss
from vpinns_jax.losses.euclidean import euclidean_loss
from vpinns_jax.losses.residual import (
    compute_fe_residual,
    compute_interpolated_fe_residual,
    compute_patch_residual,
    compute_patch_residual_terms,
)
from vpinns_jax.losses.riesz import diagonal_riesz_loss, restrict_to_free, riesz_loss

__all__ = [
    "compute_fe_residual",
    "compute_interpolated_fe_residual",
    "compute_patch_residual",
    "compute_patch_residual_terms",
    "euclidean_loss",
    "restrict_to_free",
    "riesz_loss",
    "diagonal_riesz_loss",
    "boundary_residual_loss",
]
