"""Tests for the patch residual, the residual terms, and the patch losses."""

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest
from jax_fem.problem import (
    EllipticProblem,
    constant_scalar_coefficient,
    constant_tensor_coefficient,
    constant_vector_coefficient,
    create_full_boundary_dirichlet_condition,
    create_poisson_sin_sin_problem,
)

from vpinns_jax.losses import (
    compute_patch_residual,
    compute_patch_residual_terms,
    restrict_to_free,
    riesz_loss,
)
from vpinns_jax.models import (
    ConstrainedAnsatz,
    ExactDistanceFunction,
    MultiLayerPerceptron,
    evaluate_network,
    evaluate_network_flux_divergence,
)
from vpinns_jax.test_spaces import create_linf_grid_family, create_patch_test_space


def sin_sin_solution(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.sin(jnp.pi * x[0]) * jnp.sin(jnp.pi * x[1])


def _laplace_problem(mesh) -> EllipticProblem:
    return EllipticProblem(
        diffusion=constant_tensor_coefficient(jnp.eye(2)),
        advection=constant_vector_coefficient(jnp.zeros(2)),
        reaction=constant_scalar_coefficient(0.0),
        source=constant_scalar_coefficient(0.0),
        dirichlet_conditions=(
            create_full_boundary_dirichlet_condition(
                mesh, constant_scalar_coefficient(0.0)
            ),
        ),
    )


# Remark on the normalisation: n, eta_P / |grad e| with psi_i and with phi_i,
# and eta_R / |grad e| with c_2 = 1 / pi, for the fixed error e = sin sin.
_NORMALISATION_TABLE = [
    (4, 0.884, 0.654, 1.130),
    (8, 0.910, 0.359, 0.576),
    (16, 0.916, 0.184, 0.289),
    (32, 0.918, 0.092, 0.144),
]


@pytest.mark.parametrize(
    ("cells_per_side", "normalised_ratio", "raw_ratio", "residual_ratio"),
    _NORMALISATION_TABLE,
)
def test_normalisation_table_is_reproduced(
    cells_per_side, normalised_ratio, raw_ratio, residual_ratio
) -> None:
    """With w = u - e and f = 0, the residual of e is r_i = (grad e, grad psi_i)."""
    family = create_linf_grid_family(
        np.ones((cells_per_side, cells_per_side), dtype=bool),
        spacing=1.0 / cells_per_side,
    )
    problem = _laplace_problem(family.mesh)
    error_norm = jnp.pi / jnp.sqrt(2.0)
    ratios = []
    for normalise in (True, False):
        space = create_patch_test_space(family, problem, normalise=normalise)
        trial = evaluate_network(sin_sin_solution, space.basis.physical_points)
        residual = jnp.take(
            compute_patch_residual(space, trial), family.interior_patches, axis=-2
        )
        factor = jnp.linalg.cholesky(
            restrict_to_free(space.gram, family.interior_patches)
        )
        ratios.append(jnp.sqrt(riesz_loss(residual, factor)) / error_norm)
    flux_divergence = evaluate_network_flux_divergence(
        sin_sin_solution, problem.diffusion, space.basis.physical_points
    )
    terms = compute_patch_residual_terms(space, trial, flux_divergence)
    ratios.append(jnp.sqrt(jnp.sum(terms)) / jnp.pi / error_norm)

    assert np.allclose(ratios, (normalised_ratio, raw_ratio, residual_ratio), atol=5e-4)


def test_exact_solution_has_vanishing_residual_and_terms() -> None:
    family = create_linf_grid_family(np.ones((4, 4), dtype=bool), spacing=0.25)
    problem = create_poisson_sin_sin_problem(family.mesh)
    space = create_patch_test_space(family, problem)
    points = space.basis.physical_points
    trial = evaluate_network(sin_sin_solution, points)
    terms = compute_patch_residual_terms(
        space,
        trial,
        evaluate_network_flux_divergence(sin_sin_solution, problem.diffusion, points),
    )
    residual = jnp.take(
        compute_patch_residual(space, trial), family.interior_patches, axis=-2
    )

    assert terms.shape == (family.number_of_patches, 1)
    assert jnp.allclose(terms, 0.0, atol=1e-24)
    assert jnp.max(jnp.abs(residual)) < 1e-3 * jnp.max(
        jnp.abs(space.coefficients.source)
    )


def test_patch_loss_inside_jit_matches_eager(prng_key) -> None:
    family = create_linf_grid_family(np.ones((3, 3), dtype=bool), spacing=1.0 / 3.0)
    problem = create_poisson_sin_sin_problem(family.mesh)
    space = create_patch_test_space(family, problem)
    points = space.basis.physical_points
    factor = jnp.linalg.cholesky(restrict_to_free(space.gram, family.interior_patches))
    distance = ExactDistanceFunction("unit_square")

    def loss_fn(model: MultiLayerPerceptron) -> jnp.ndarray:
        trial_network = ConstrainedAnsatz(model, distance)
        trial = evaluate_network(trial_network, points)
        flux_divergence = evaluate_network_flux_divergence(
            trial_network, problem.diffusion, points
        )
        residual = jnp.take(
            compute_patch_residual(space, trial), family.interior_patches, axis=-2
        )
        terms = compute_patch_residual_terms(space, trial, flux_divergence)
        return riesz_loss(residual, factor) + jnp.sum(terms)

    network = MultiLayerPerceptron(width=8, depth=2, key=prng_key)
    assert jnp.allclose(eqx.filter_jit(loss_fn)(network), loss_fn(network))
    gradient = eqx.filter_grad(loss_fn)(network)
    assert jnp.all(jnp.isfinite(gradient.net.layers[0].weight))
