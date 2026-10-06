"""
Evaluation of scalar networks at batches of points.
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
from jax_fem.function import FiniteElementFunctionEvaluation
from jax_fem.problem import TensorCoefficient

#: A scalar function of one point, ``x: (d,) -> ()``, such as a network.
type Network = Callable[[jax.Array], jax.Array]


def evaluate_network_values(network: Network, points: jax.Array) -> jax.Array:
    """
    Values of a scalar network at points.

    Parameters
    ----------
    network : Network
        Scalar function of one point.
    points : jax.Array
        Points as row vectors, shape ``(..., 1, d)``.

    Returns
    -------
    values : jax.Array
        Network values, shape ``(..., 1, 1)``.
    """
    values = jnp.vectorize(network, signature="(d)->()")(points)
    return jnp.expand_dims(values, axis=-1)


def evaluate_network(
    network: Network, points: jax.Array
) -> FiniteElementFunctionEvaluation:
    """
    Values and gradients of a scalar network at points.

    Parameters
    ----------
    network : Network
        Scalar function of one point.
    points : jax.Array
        Points as row vectors, shape ``(..., 1, d)``.

    Returns
    -------
    evaluation : FiniteElementFunctionEvaluation
        Values, shape ``(..., 1, 1)``, and gradients, shape ``(..., 1, d)``;
        ``divergence`` is ``None``.

    Notes
    -----
    The result has the layout of a scalar finite element function evaluated
    at quadrature points. It can therefore be passed as the trial argument of
    the ``jax_fem`` forms, which read only ``values`` and ``gradients``, and
    as a field of the ``jax_fem`` error functionals.
    """
    values, gradients = jnp.vectorize(
        jax.value_and_grad(network), signature="(d)->(),(d)"
    )(points)
    return FiniteElementFunctionEvaluation(
        values=jnp.expand_dims(values, axis=-1),
        gradients=gradients,
        divergence=None,
    )


def evaluate_network_flux_divergence(
    network: Network, diffusion: TensorCoefficient, points: jax.Array
) -> jax.Array:
    """
    Divergence of the diffusive flux of a scalar network at points.

    Parameters
    ----------
    network : Network
        Scalar function of one point.
    diffusion : TensorCoefficient
        Diffusion tensor ``A``; takes points of shape ``(*batch, 1, d)`` and
        returns ``(*batch, d, d)``.
    points : jax.Array
        Points as row vectors, shape ``(..., 1, d)``.

    Returns
    -------
    flux_divergence : jax.Array
        ``div(A grad w)``, shape ``(..., 1, 1)``.

    Notes
    -----
    The flux ``A grad w`` is differentiated by forward-mode automatic
    differentiation, including the derivative of ``A`` itself, so a variable
    diffusion contributes ``grad(A) : grad w`` as well as ``A : hess w``.
    """

    def flux(x: jax.Array) -> jax.Array:
        """
        Diffusive flux at one point.

        Parameters
        ----------
        x : jax.Array
            Point, shape ``(d,)``.

        Returns
        -------
        flux : jax.Array
            ``A(x) grad w(x)``, shape ``(d,)``.
        """
        return diffusion(jnp.expand_dims(x, axis=-2)) @ jax.grad(network)(x)

    def divergence(x: jax.Array) -> jax.Array:
        """
        Divergence of the flux at one point.

        Parameters
        ----------
        x : jax.Array
            Point, shape ``(d,)``.

        Returns
        -------
        divergence : jax.Array
            ``div(A grad w)(x)``, shape ``()``.
        """
        return jnp.trace(jax.jacfwd(flux)(x))

    values = jnp.vectorize(divergence, signature="(d)->()")(points)
    return jnp.expand_dims(values, axis=-1)
