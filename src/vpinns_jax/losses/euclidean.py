"""
Euclidean loss of a residual vector.
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
