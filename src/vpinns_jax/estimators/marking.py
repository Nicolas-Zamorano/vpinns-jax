"""Marking rules for adaptive refinement (math reference Section 4.4)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax_fem.mesh import mark_cells_by_dorfler_bulk_criterion

__all__ = ["mark_by_dorfler_bulk_criterion", "cap_marked_fraction"]


def mark_by_dorfler_bulk_criterion(indicators: jax.Array, theta: float = 0.5) -> jax.Array:
    r"""Doerfler bulk marking: mark the smallest set with ``sum_M eta_i^2 >= theta sum eta_i^2``.

    Thin re-export of jax-fem's own ``mark_cells_by_dorfler_bulk_criterion``,
    which is already generic over any per-index-set indicator array -- it
    applies unchanged to patch indicators, not only cell indicators.
    """
    return mark_cells_by_dorfler_bulk_criterion(indicators, theta)


def cap_marked_fraction(
    marked: jax.Array, indicators: jax.Array, max_fraction: float
) -> jax.Array:
    """Cap the number of marked entries at ``max_fraction * len(indicators)``.

    Controls refinement speed: keeps the largest-indicator entries among
    those already marked, dropping the rest.

    Parameters
    ----------
    marked : jax.Array
        Boolean mask, shape ``(N,)`` (e.g. from ``mark_by_dorfler_bulk_criterion``).
    indicators : jax.Array
        The same indicators ``marked`` was computed from, shape ``(N,)``.
    max_fraction : float
        Maximum marked fraction, in ``(0, 1]``.

    Returns
    -------
    jax.Array
        Boolean mask, shape ``(N,)``, ``marked`` with excess entries dropped.
    """
    if not (0.0 < max_fraction <= 1.0):
        raise ValueError(f"max_fraction must be in (0, 1], got {max_fraction}.")
    max_count = max(1, int(max_fraction * indicators.shape[0]))
    order = jnp.argsort(jnp.where(marked, indicators, -jnp.inf))[::-1]
    rank = jnp.argsort(order)
    return marked & (rank < max_count)
