"""Tests for the patch families integrated patch by patch."""

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest
from jax_fem.assembly import integrate_cellwise
from jax_fem.problem import create_poisson_sin_sin_problem

from vpinns_jax.losses import (
    compute_meshfree_patch_residual,
    compute_meshfree_patch_residual_terms,
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
from vpinns_jax.test_spaces import (
    create_bilinear_family,
    create_crisscross_family,
    create_linf_grid_family,
    create_linf_vertex_family,
    create_meshfree_patch_test_space,
    create_patch_test_space,
    create_vertex_star_family,
)

_CELLS_PER_SIDE = 4
_SPACING = 1.0 / _CELLS_PER_SIDE


def sin_sin_solution(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.sin(jnp.pi * x[0]) * jnp.sin(jnp.pi * x[1])


def _boundary_sides(family) -> np.ndarray:
    """Boundary lines through each centre: 0 inside, 1 on a side, 2 at a corner."""
    centres = np.asarray(family.centres)
    return np.sum(np.isclose(centres, 0.0) | np.isclose(centres, 1.0), axis=-1)


def _weight_integrals(space) -> np.ndarray:
    return np.asarray(integrate_cellwise(space.raw_weights.values, space.weights))[
        :, 0, 0, 0
    ]


@pytest.fixture(scope="module")
def problem(unit_square_mesh):
    return create_poisson_sin_sin_problem(unit_square_mesh)


def test_crisscross_family_configurations(problem) -> None:
    n = _CELLS_PER_SIDE
    family = create_crisscross_family(n)
    space = create_meshfree_patch_test_space(family, problem)
    grid_points = (n + 1) ** 2
    sides = _boundary_sides(family)[:grid_points]

    assert family.number_of_patches == grid_points + n**2
    assert family.interior_patches.shape == ((n - 1) ** 2 + n**2,)
    assert np.count_nonzero(family.boundary_patch_mask) == 4 * n
    assert space.covering_bounds == pytest.approx((1.0, 1.0), abs=1e-14)
    # Interior grid point, point on a side, corner; then the cell centres.
    assert np.allclose(
        family.patch_diameters[:grid_points],
        _SPACING * np.array([2.0, 2.0, np.sqrt(2.0)])[sides],
    )
    assert np.allclose(family.patch_diameters[grid_points:], _SPACING * np.sqrt(2.0))
    integrals = _weight_integrals(space) / _SPACING**2
    assert np.allclose(integrals[:grid_points], np.array([2.0, 1.0, 0.5])[sides] / 3)
    assert np.allclose(integrals[grid_points:], 1.0 / 3.0)


@pytest.mark.parametrize(
    ("create_family", "expected_integrals", "expected_covering_bounds"),
    [
        (create_bilinear_family, (1.0, 0.5, 0.25), (1.0, 1.0)),
        (create_linf_vertex_family, (4.0 / 3.0, 2.0 / 3.0, 1.0 / 3.0), (1.0, 2.0)),
    ],
)
def test_grid_point_family_configurations(
    create_family, expected_integrals, expected_covering_bounds, problem
) -> None:
    n = _CELLS_PER_SIDE
    family = create_family(n)
    space = create_meshfree_patch_test_space(family, problem)
    sides = _boundary_sides(family)

    assert family.number_of_patches == (n + 1) ** 2
    assert family.interior_patches.shape == ((n - 1) ** 2,)
    assert np.count_nonzero(family.boundary_patch_mask) == 4 * n
    assert space.covering_bounds == pytest.approx(expected_covering_bounds, abs=1e-14)
    # Interior grid point, point on a side, corner.
    assert np.allclose(
        family.patch_diameters,
        _SPACING * np.array([2.0 * np.sqrt(2.0), np.sqrt(5.0), np.sqrt(2.0)])[sides],
    )
    assert np.allclose(
        _weight_integrals(space), _SPACING**2 * np.array(expected_integrals)[sides]
    )


@pytest.mark.parametrize(
    ("create_family", "normalise"),
    [
        (create_crisscross_family, False),
        (create_bilinear_family, False),
        (create_linf_vertex_family, True),
    ],
)
def test_test_functions_are_a_partition_of_unity(
    create_family, normalise, problem
) -> None:
    """sum_j psi_j = 1 makes every row of the full Gram matrix sum to zero."""
    space = create_meshfree_patch_test_space(
        create_family(_CELLS_PER_SIDE), problem, normalise=normalise
    )
    gram = space.gram.todense()
    interior = space.family.interior_patches

    assert jnp.allclose(gram, gram.T, atol=1e-14)
    assert jnp.allclose(jnp.sum(gram, axis=-1), 0.0, atol=1e-13)
    assert jnp.all(jnp.linalg.eigvalsh(restrict_to_free(space.gram, interior)) > 0.0)


def test_crisscross_gram_matrix_is_the_rotated_five_point_stencil(problem) -> None:
    n = _CELLS_PER_SIDE
    family = create_crisscross_family(n)
    interior = np.asarray(family.interior_patches)
    gram = np.asarray(
        restrict_to_free(
            create_meshfree_patch_test_space(family, problem).gram, interior
        )
    )
    centres = np.asarray(family.centres)[interior]
    offsets = np.abs(
        np.expand_dims(centres, axis=-2) - np.expand_dims(centres, axis=-3)
    )
    adjacent = np.all(np.isclose(offsets, _SPACING / 2), axis=-1)

    assert np.allclose(np.diagonal(gram), 4.0)
    assert np.allclose(gram, 4.0 * np.eye(interior.shape[0]) - adjacent)


def test_bilinear_gram_matrix_is_the_q1_stiffness(problem) -> None:
    family = create_bilinear_family(_CELLS_PER_SIDE)
    interior = np.asarray(family.interior_patches)
    gram = np.asarray(
        restrict_to_free(
            create_meshfree_patch_test_space(family, problem).gram, interior
        )
    )
    centres = np.asarray(family.centres)[interior]
    offsets = np.max(
        np.abs(np.expand_dims(centres, axis=-2) - np.expand_dims(centres, axis=-3)),
        axis=-1,
    )
    neighbours = np.isclose(offsets, _SPACING)

    assert np.allclose(gram, 8.0 / 3.0 * np.eye(interior.shape[0]) - neighbours / 3.0)


def _compare_with_mesh_based_space(
    meshfree_space, mesh_space, problem, prng_key
) -> None:
    network = ConstrainedAnsatz(
        MultiLayerPerceptron(width=8, depth=2, key=prng_key),
        ExactDistanceFunction("unit_square"),
    )

    def evaluate(points):
        return (
            evaluate_network(network, points),
            evaluate_network_flux_divergence(network, problem.diffusion, points),
        )

    meshfree_trial, meshfree_divergence = evaluate(meshfree_space.points)
    mesh_trial, mesh_divergence = evaluate(mesh_space.basis.physical_points)

    assert jnp.allclose(meshfree_space.gram.todense(), mesh_space.gram.todense())
    assert jnp.allclose(
        compute_meshfree_patch_residual(meshfree_space, meshfree_trial),
        compute_patch_residual(mesh_space, mesh_trial),
    )
    assert jnp.allclose(
        compute_meshfree_patch_residual_terms(
            meshfree_space, meshfree_trial, meshfree_divergence
        ),
        compute_patch_residual_terms(mesh_space, mesh_trial, mesh_divergence),
    )


def test_linf_vertex_family_matches_the_mesh_based_family(problem, prng_key) -> None:
    """The local triangles are the cells of the mesh-based family's integration mesh."""
    n = _CELLS_PER_SIDE
    mesh_family = create_linf_grid_family(np.ones((n, n), dtype=bool), spacing=_SPACING)
    meshfree_family = create_linf_vertex_family(n)

    assert jnp.allclose(meshfree_family.patch_diameters, mesh_family.patch_diameters)
    assert jnp.array_equal(
        meshfree_family.boundary_patch_mask, mesh_family.boundary_patch_mask
    )
    _compare_with_mesh_based_space(
        create_meshfree_patch_test_space(meshfree_family, problem, normalise=True),
        create_patch_test_space(mesh_family, problem, quadrature_degree=2),
        problem,
        prng_key,
    )


def test_crisscross_family_matches_vertex_stars_of_the_crisscross_mesh(
    problem, prng_key
) -> None:
    n = _CELLS_PER_SIDE
    crisscross_mesh = create_linf_grid_family(
        np.ones((n, n), dtype=bool), spacing=_SPACING
    ).mesh
    mesh_family = create_vertex_star_family(crisscross_mesh)
    meshfree_family = create_crisscross_family(n)

    assert jnp.allclose(
        meshfree_family.centres, crisscross_mesh.vertex_coordinates[:, 0]
    )
    assert jnp.allclose(meshfree_family.patch_diameters, mesh_family.patch_diameters)
    _compare_with_mesh_based_space(
        create_meshfree_patch_test_space(meshfree_family, problem),
        create_patch_test_space(mesh_family, problem, quadrature_degree=2),
        problem,
        prng_key,
    )


@pytest.mark.parametrize(
    ("create_family", "normalise"),
    [
        (create_crisscross_family, False),
        (create_bilinear_family, False),
        (create_linf_vertex_family, True),
    ],
)
def test_exact_solution_has_vanishing_residual_terms(
    create_family, normalise, problem
) -> None:
    space = create_meshfree_patch_test_space(
        create_family(8), problem, normalise=normalise, quadrature_degree=5
    )
    trial = evaluate_network(sin_sin_solution, space.points)
    terms = compute_meshfree_patch_residual_terms(
        space,
        trial,
        evaluate_network_flux_divergence(
            sin_sin_solution, problem.diffusion, space.points
        ),
    )
    residual = jnp.take(
        compute_meshfree_patch_residual(space, trial),
        space.family.interior_patches,
        axis=-2,
    )

    assert terms.shape == (space.family.number_of_patches, 1)
    assert jnp.allclose(terms, 0.0, atol=1e-24)
    assert jnp.max(jnp.abs(residual)) < 1e-3 * jnp.max(
        jnp.abs(space.coefficients.source)
    )


def test_meshfree_patch_loss_inside_jit_matches_eager(problem, prng_key) -> None:
    space = create_meshfree_patch_test_space(create_crisscross_family(3), problem)
    interior = space.family.interior_patches
    factor = jnp.linalg.cholesky(restrict_to_free(space.gram, interior))
    distance = ExactDistanceFunction("unit_square")

    def loss_fn(model: MultiLayerPerceptron) -> jnp.ndarray:
        trial_network = ConstrainedAnsatz(model, distance)
        trial = evaluate_network(trial_network, space.points)
        flux_divergence = evaluate_network_flux_divergence(
            trial_network, problem.diffusion, space.points
        )
        residual = jnp.take(
            compute_meshfree_patch_residual(space, trial), interior, axis=-2
        )
        terms = compute_meshfree_patch_residual_terms(space, trial, flux_divergence)
        return riesz_loss(residual, factor) + jnp.sum(terms)

    network = MultiLayerPerceptron(width=8, depth=2, key=prng_key)
    assert jnp.allclose(eqx.filter_jit(loss_fn)(network), loss_fn(network))
    gradient = eqx.filter_grad(loss_fn)(network)
    assert jnp.all(jnp.isfinite(gradient.net.layers[0].weight))
