"""
Euclidean losses of residual vectors.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def euclidean_loss(residual: jax.Array) -> jax.Array:
    """
    Sum of squared residual entries.

    Parameters
    ----------
    residual : jax.Array
        Residual vector, already restricted to the test functions, e.g.
        shape ``(N_free, 1)``.

    Returns
    -------
    loss : jax.Array
        ``r^T r``, shape ``()``.
    """
    return jnp.sum(residual**2)


def mf_vpinn_loss(free_residual: jax.Array, patch_areas: jax.Array) -> jax.Array:
    """
    Loss of the mesh-free VPINN of Berrone and Pintore (2024).

    Parameters
    ----------
    free_residual : jax.Array
        Residual at the test functions, shape ``(N_free, 1)``.
    patch_areas : jax.Array
        Area ``|P_i|`` of the patch of each test function, shape
        ``(N_free,)``.

    Returns
    -------
    loss : jax.Array
        ``L_MF = (1 / N_free) sum_i r_i^2 / |P_i|``, shape ``()``.

    Notes
    -----
    The factor ``1 / |P_i|`` gives the same importance to every patch. The
    loss is not a dual norm of the residual, so it does not control the
    energy error.
    """
    return jnp.mean(free_residual**2 / jnp.expand_dims(patch_areas, axis=-1))
