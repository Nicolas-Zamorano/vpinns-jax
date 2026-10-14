"""
Errors of network trial functions against exact solutions.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax_fem.assembly import integrate_cellwise
from jax_fem.forms import Functional
from jax_fem.problem import EllipticProblem
from jax_fem.space import CellBasis

from vpinns_jax.models.evaluation import Network, evaluate_network


def compute_network_error(
    network: Network,
    basis: CellBasis,
    problem: EllipticProblem,
    functional: Functional,
    reduce: bool = True,
) -> jax.Array:
    """
    Error norm of a network for a pointwise squared-error functional.

    Parameters
    ----------
    network : Network
        Scalar trial function of one point.
    basis : CellBasis
        Evaluated basis whose quadrature defines the integration.
    problem : EllipticProblem
        Problem providing the exact solution and gradient.
    functional : Functional
        Squared-error functional, e.g. ``l2_error_functional``,
        ``h1_seminorm_error_functional``, or ``energy_error_functional`` from
        ``jax_fem.diagnostics``.
    reduce : bool
        Return the global norm if ``True``, per-cell norms otherwise.

    Returns
    -------
    error : jax.Array
        Global error, shape ``()``, or per-cell errors, shape ``(K, 1, 1, 1)``.

    Notes
    -----
    The network is evaluated at ``basis.physical_points`` and passed to the
    functional as the ``"solution"`` field, so the error is computed exactly
    as for a finite element solution, including the tangential projection of
    the exact gradient on embedded meshes. The basis is used only as a
    quadrature grid; the network need not belong to its space.
    """
    fields = {"solution": evaluate_network(network, basis.physical_points)}
    local_errors = integrate_cellwise(
        functional(basis, fields, problem), basis.physical_weights
    )
    if reduce:
        return jnp.sqrt(jnp.sum(local_errors))
    return jnp.sqrt(local_errors)


def _zero(x: jax.Array) -> jax.Array:
    """
    Zero function.

    Parameters
    ----------
    x : jax.Array
        Point, shape ``(d,)``.

    Returns
    -------
    value : jax.Array
        Zero, shape ``()``.
    """
    return jnp.zeros((), dtype=x.dtype)


def compute_solution_norm(
    basis: CellBasis, problem: EllipticProblem, functional: Functional
) -> jax.Array:
    """
    Norm of the exact solution measured by a squared-error functional.

    Parameters
    ----------
    basis : CellBasis
        Evaluated basis whose quadrature defines the integration.
    problem : EllipticProblem
        Problem providing the exact solution and gradient.
    functional : Functional
        Squared-error functional, e.g. ``energy_error_functional`` for
        ``|u|_A``.

    Returns
    -------
    norm : jax.Array
        The error of the zero function, i.e. the norm of ``u``, shape ``()``.
    """
    return compute_network_error(_zero, basis, problem, functional)
