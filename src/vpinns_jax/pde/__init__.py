"""PDE problems and boundary formulations (implementation plan Component 1)."""

from vpinns_jax.pde.advection_diffusion import (
    create_advection_diffusion_problem,
    create_simple_elliptic_problem,
)
from vpinns_jax.pde.base import (
    DirichletCondition,
    EllipticPDE,
    EllipticProblem,
    create_full_boundary_dirichlet_condition,
    energy_reaction_coefficient,
    evaluate_elliptic_coefficients,
)
from vpinns_jax.pde.poisson import (
    create_poisson_cos_sin_problem,
    create_poisson_exponential_problem,
    create_poisson_l_shaped_singular_problem,
    create_poisson_singular_problem,
    create_poisson_sin_sin_problem,
)

__all__ = [
    "EllipticPDE",
    "EllipticProblem",
    "DirichletCondition",
    "create_full_boundary_dirichlet_condition",
    "evaluate_elliptic_coefficients",
    "energy_reaction_coefficient",
    "create_simple_elliptic_problem",
    "create_advection_diffusion_problem",
    "create_poisson_sin_sin_problem",
    "create_poisson_cos_sin_problem",
    "create_poisson_exponential_problem",
    "create_poisson_singular_problem",
    "create_poisson_l_shaped_singular_problem",
]
