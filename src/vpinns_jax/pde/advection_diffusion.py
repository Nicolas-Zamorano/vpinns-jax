"""General second-order advection-diffusion-reaction problems.

Non-symmetric bilinear form ``a(w, v)`` (math reference Section 1); the
symmetric energy inner product ``(v, w)_A`` is obtained separately via
``pde.base.energy_reaction_coefficient``.
"""

from __future__ import annotations

from collections.abc import Sequence

from jax_fem.problem import DirichletCondition, create_simple_elliptic_problem

from vpinns_jax.pde.base import (
    EllipticPDE,
    ScalarCoefficient,
    TensorCoefficient,
    VectorCoefficient,
)

__all__ = [
    "create_simple_elliptic_problem",
    "create_advection_diffusion_problem",
]


def create_advection_diffusion_problem(
    *,
    name: str,
    diffusion: TensorCoefficient,
    advection: VectorCoefficient,
    reaction: ScalarCoefficient,
    source: ScalarCoefficient,
    dirichlet_conditions: Sequence[DirichletCondition],
    exact_solution: ScalarCoefficient | None = None,
    exact_gradient: VectorCoefficient | None = None,
    advection_divergence: ScalarCoefficient | None = None,
) -> EllipticPDE:
    r"""Build a general ``EllipticPDE`` from user-supplied coefficient callables.

    ``L u = -div(mu grad u) + beta . grad u + sigma u = f`` (math reference
    Section 1). Well-posedness requires ``mu >= mu_0 > 0`` and
    ``sigma - div(beta)/2 >= 0`` a.e. in the domain; this constructor does
    not check either condition, it only assembles the data.

    Parameters
    ----------
    name : str
        Problem name, used e.g. as an output-directory stem by plotting
        helpers.
    diffusion, advection, reaction, source : coefficient callables
        ``mu(x)``, ``beta(x)``, ``sigma(x)``, ``f(x)``.
    dirichlet_conditions : Sequence[DirichletCondition]
        Boundary tags and their Dirichlet data.
    exact_solution, exact_gradient : coefficient callables | None
        Optional, for manufactured-solution error diagnostics.
    advection_divergence : ScalarCoefficient | None
        ``div(beta)(x)``, optional; only needed for the energy-norm error
        diagnostic and the Riesz-loss energy inner product
        (``pde.base.energy_reaction_coefficient``).

    Returns
    -------
    EllipticPDE
    """
    return EllipticPDE(
        name=name,
        diffusion=diffusion,
        advection=advection,
        reaction=reaction,
        source=source,
        dirichlet_conditions=tuple(dirichlet_conditions),
        exact_solution=exact_solution,
        exact_gradient=exact_gradient,
        advection_divergence=advection_divergence,
    )
