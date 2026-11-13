"""
Checks of the two-sided error bounds along a training.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from vpinns_jax.analysis.monitoring import TrainingRecord


class ErrorBound(NamedTuple):
    """
    Two-sided bound ``eta <= |u - w|_A <= a eta + b rho`` of the paper.

    Attributes
    ----------
    residual_term : str
        Name of ``eta^2`` among the recorded loss terms or estimators.
    estimator : str
        Name of ``rho^2`` among the recorded loss terms or estimators.
    quasi_interpolant_norm : float
        ``a``: the norm ``||Pi||`` of the quasi-interpolant, its computed
        value ``||Pi||_h``, or the a priori bound ``N_ov K``.
    remainder_constant : float
        ``b = sqrt(S_max) / S_min``, one for a partition of unity.

    Notes
    -----
    For the normalised ``l^inf`` family, the main theorem of the paper gives
    ``eta_psi <= |u - w|_A <= ||Pi_psi|| eta_psi + sqrt(2) rho``; for a
    family whose weights sum to one, its corollary for partitions of unity
    gives the same with ``eta_P``, ``Pi_P`` and ``b = 1``. The lower bound
    holds exactly, so a computed ratio ``eta / |u - w|_A`` above one reveals
    a quadrature error.
    """

    residual_term: str
    estimator: str
    quasi_interpolant_norm: float
    remainder_constant: float


class BoundRatios(NamedTuple):
    """
    The two sides of an error bound divided by the error.

    Attributes
    ----------
    lower : jax.Array
        ``eta / |u - w|_A``, at most one by the lower bound, shape
        ``(...,)``.
    upper : jax.Array
        ``(a eta + b rho) / |u - w|_A``, at least one by the upper bound,
        shape ``(...,)``.
    """

    lower: jax.Array
    upper: jax.Array


def compute_bound_ratios(bound: ErrorBound, record: TrainingRecord) -> BoundRatios:
    """
    Sides of a two-sided error bound relative to the error of a record.

    Parameters
    ----------
    bound : ErrorBound
        The bound and the names of its terms.
    record : TrainingRecord
        Record of one step, or the stacked records of a ``TrainingHistory``,
        whose loss terms or estimators contain ``eta^2`` and ``rho^2``.

    Returns
    -------
    ratios : BoundRatios
        ``eta / |u - w|_A`` and ``(a eta + b rho) / |u - w|_A``, with the
        shape of ``record.energy_error``.

    Raises
    ------
    KeyError
        If the record contains neither term under its name.
    """
    recorded = {**record.terms, **record.estimators}
    residual_term = jnp.sqrt(recorded[bound.residual_term])
    estimator = jnp.sqrt(recorded[bound.estimator])
    error = record.energy_error
    return BoundRatios(
        lower=residual_term / error,
        upper=(
            bound.quasi_interpolant_norm * residual_term
            + bound.remainder_constant * estimator
        )
        / error,
    )
