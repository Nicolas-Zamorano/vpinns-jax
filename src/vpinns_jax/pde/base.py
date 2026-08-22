"""``EllipticPDE``: vpinns-jax's PDE data model, built directly on jax-fem.

Math reference Section 1: the model problem is

    L u := -div(mu grad u) + beta . grad u + sigma u = f   in Omega,
    u = 0                                                   on Gamma,

with energy inner product ``(v, w)_A = int mu grad(v).grad(w) + (sigma -
div(beta)/2) v w``.

vpinns-jax does not duplicate jax-fem's PDE data model with a second,
driftable dataclass: ``EllipticPDE`` *is*
``jax_fem.problem.elliptic.EllipticProblem``, re-exported here so the rest
of vpinns-jax can import PDE types from one place. Field correspondence
(math reference notation -> ``EllipticProblem`` field):

    mu    (diffusion tensor A)      -> problem.diffusion
    beta  (advection velocity)      -> problem.advection
    sigma (reaction coefficient)    -> problem.reaction
    f     (source term)             -> problem.source
    g     (Dirichlet data)          -> problem.dirichlet_conditions[i].value

This module adds the one piece jax-fem's ``EllipticProblem`` does not
provide directly: the symmetrized reaction coefficient the energy inner
product needs.
"""

from __future__ import annotations

import jax
from jax_fem.problem import (
    DirichletCondition,
    EllipticCoefficientValues,
    EllipticProblem,
    ScalarCoefficient,
    TensorCoefficient,
    VectorCoefficient,
    constant_scalar_coefficient,
    constant_tensor_coefficient,
    constant_vector_coefficient,
    create_full_boundary_dirichlet_condition,
    evaluate_elliptic_coefficients,
    isotropic_tensor_coefficient,
)

#: vpinns-jax's PDE type: an alias for jax-fem's ``EllipticProblem``.
EllipticPDE = EllipticProblem

__all__ = [
    "EllipticPDE",
    "EllipticProblem",
    "EllipticCoefficientValues",
    "ScalarCoefficient",
    "VectorCoefficient",
    "TensorCoefficient",
    "DirichletCondition",
    "create_full_boundary_dirichlet_condition",
    "evaluate_elliptic_coefficients",
    "constant_scalar_coefficient",
    "constant_vector_coefficient",
    "constant_tensor_coefficient",
    "isotropic_tensor_coefficient",
    "energy_reaction_coefficient",
]


def energy_reaction_coefficient(problem: EllipticPDE) -> ScalarCoefficient:
    r"""``(sigma - div(beta)/2)(x)``: the energy inner product's reaction coefficient.

    The energy inner product (math reference Section 1) is

    .. math::

        (v, w)_A = \int_\Omega \mu \nabla v \cdot \nabla w
            + \Big(\sigma - \tfrac12 \nabla\cdot\beta\Big) v w \, dx,

    i.e. it has *no* advection term (unlike the bilinear form ``a(w, v)``,
    which is generally non-symmetric) and a *symmetrized* reaction
    coefficient. This function evaluates that coefficient; it is what
    ``losses.riesz``'s energy Gram matrix / patch-energy assembly needs.

    Parameters
    ----------
    problem : EllipticPDE
        Requires ``problem.advection_divergence`` (``div(beta)``).

    Returns
    -------
    ScalarCoefficient
        ``x -> (sigma - div(beta)/2)(x)``, shape ``(*batch, 1, 1)``.
    """
    if problem.advection_divergence is None:
        raise ValueError(
            "energy_reaction_coefficient requires problem.advection_divergence "
            "(needed for the symmetrized reaction coefficient sigma - div(beta)/2)."
        )
    reaction = problem.reaction
    advection_divergence = problem.advection_divergence

    def coefficient(points: jax.Array) -> jax.Array:
        return reaction(points) - 0.5 * advection_divergence(points)

    return coefficient
