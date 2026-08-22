"""Euclidean loss ``L_E(w_theta) = sum_i |r_h(w_theta, psi_i)|^2`` (math reference Section 2.3)."""

from __future__ import annotations

import jax
import jax.numpy as jnp

__all__ = ["euclidean_loss"]


def euclidean_loss(residual: jax.Array) -> jax.Array:
    """``L_E = r^T r``, over whatever index set ``residual`` already restricts to.

    Callers pre-restrict: e.g. ``residual[fe_test_space.free_dofs, 0]`` for
    an FE test space, or the full ``(N_patch,)`` vector for a patch test
    space (every patch center is already interior by construction).
    """
    return jnp.sum(residual**2)
