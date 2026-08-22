"""Poisson benchmark problems (``mu = I``, ``beta = 0``, ``sigma = 0``).

Re-exported from jax-fem's manufactured-problem factories
(``jax_fem.problem.manufactured``), which already provide exactly the
benchmark family the implementation plan calls for: a smooth sine/cosine
solution and corner-singularity solutions (a mild one at a convex unit-square
corner, and a genuine re-entrant-corner singularity on the L-shaped domain).
vpinns-jax does not re-derive these manufactured source terms; it names them
here so PDE problems can be imported from one place (``vpinns_jax.pde``).
"""

from __future__ import annotations

from jax_fem.problem import (
    create_poisson_cos_sin_problem,
    create_poisson_exponential_problem,
    create_poisson_l_shaped_singular_problem,
    create_poisson_singular_problem,
    create_poisson_sin_sin_problem,
)

__all__ = [
    "create_poisson_sin_sin_problem",
    "create_poisson_cos_sin_problem",
    "create_poisson_exponential_problem",
    "create_poisson_singular_problem",
    "create_poisson_l_shaped_singular_problem",
]
