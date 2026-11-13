"""Tests for the criss-cross meshes, the constants of (W2) and (W3), the norms
of the quasi-interpolants, and the bound checks."""

import jax.numpy as jnp
import numpy as np
import pytest
from jax_fem.problem import create_poisson_sin_sin_problem

from vpinns_jax.analysis import (
    ErrorBound,
    TrainingRecord,
    compute_bound_ratios,
    compute_quasi_interpolant_norm,
    create_training_monitor,
)
from vpinns_jax.test_spaces import (
    compute_linf_poincare_constants,
    create_crisscross_family,
    create_crisscross_mesh,
    create_crisscross_unit_square_mesh,
    create_linf_vertex_family,
)
from vpinns_jax.test_spaces.poincare_constants import round_up


def _triangles(mesh) -> np.ndarray:
    vertices = np.asarray(mesh.vertex_coordinates)[:, 0, :]
    return vertices[np.asarray(mesh.cells_to_vertices)]


def test_crisscross_mesh_cuts_every_cell_along_both_diagonals() -> None:
    mesh = create_crisscross_mesh((-1.0, 0.0), (1.0, 1.0), (4, 2))
    triangles = _triangles(mesh)
    areas = 0.5 * np.linalg.det(triangles[:, 1:] - triangles[:, :1])

    assert mesh.vertex_coordinates.shape[0] == 5 * 3 + 4 * 2
    assert triangles.shape == (4 * 2 * 4, 3, 2)
    # Counterclockwise quarters of cells of size 0.5 x 0.5.
    assert np.allclose(areas, 0.0625)
    assert set(mesh.boundary_tag_names) == {"bottom", "right", "top", "left"}


def test_crisscross_mesh_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="cells"):
        create_crisscross_mesh((0.0, 0.0), (1.0, 1.0), (0, 2))
    with pytest.raises(ValueError, match="upper"):
        create_crisscross_mesh((0.0, 0.0), (0.0, 1.0), (1, 1))


@pytest.mark.parametrize(
    "create_family", [create_linf_vertex_family, create_crisscross_family]
)
def test_family_weights_are_affine_on_the_crisscross_triangles(create_family) -> None:
    """An affine function equals the mean of its vertex values at the centroid."""
    family = create_family(4)
    triangles = _triangles(create_crisscross_unit_square_mesh(8))
    weight = jnp.vectorize(family.weight, signature="(d),(d),()->()")
    centres = jnp.expand_dims(family.centres, axis=-2)
    kinds = jnp.expand_dims(family.kinds, axis=-1)

    at_vertices = weight(
        jnp.asarray(triangles),
        jnp.expand_dims(centres, axis=-2),
        jnp.expand_dims(kinds, axis=-1),
    )
    at_centroids = weight(jnp.asarray(triangles.mean(axis=-2)), centres, kinds)

    assert np.allclose(at_centroids, at_vertices.mean(axis=-1), atol=1e-12)


@pytest.mark.parametrize(
    ("configuration", "expected"),
    [
        ("interior", 2.0 / np.pi),
        ("side", 2.0 / np.pi),
        ("corner", np.sqrt(2.0) / np.pi),
    ],
)
def test_unweighted_constants_approach_the_closed_forms_from_below(
    configuration, expected
) -> None:
    constants = compute_linf_poincare_constants(configuration, 16)

    assert expected - 1e-3 < constants.unweighted <= expected


@pytest.mark.parametrize(
    ("configuration", "expected"),
    [("interior", 0.4691), ("side", 0.4691), ("corner", 0.3607)],
)
def test_weighted_constants_increase_under_refinement(configuration, expected) -> None:
    coarse = compute_linf_poincare_constants(configuration, 8).weighted
    fine = compute_linf_poincare_constants(configuration, 16).weighted

    assert coarse <= fine <= expected + 1e-4
    assert fine == pytest.approx(expected, abs=5e-4)


def test_side_and_interior_patches_share_the_weighted_constant() -> None:
    """The odd reflection of the side patch is the interior patch."""
    interior = compute_linf_poincare_constants("interior", 8)
    side = compute_linf_poincare_constants("side", 8)

    assert side.weighted == pytest.approx(interior.weighted, rel=1e-8)
    assert side.diameter == pytest.approx(np.sqrt(5.0))
    assert interior.diameter == pytest.approx(2.0 * np.sqrt(2.0))


def test_unknown_configuration_is_rejected() -> None:
    with pytest.raises(ValueError, match="configuration"):
        compute_linf_poincare_constants("edge", 8)


@pytest.mark.parametrize(
    ("value", "expected"), [(0.469101, 0.470), (0.3606, 0.361), (0.469, 0.469)]
)
def test_round_up(value, expected) -> None:
    assert round_up(value, 3) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("create_family", "normalise", "expected"),
    [
        (create_linf_vertex_family, True, 0.4781),
        (create_crisscross_family, False, 0.7608),
    ],
)
def test_quasi_interpolant_norms_of_the_coarsest_families(
    create_family, normalise, expected
) -> None:
    problem = create_poisson_sin_sin_problem(create_crisscross_unit_square_mesh(2))
    family = create_family(2)
    coarse = compute_quasi_interpolant_norm(
        family, problem, 2, normalise=normalise, refinement=2
    )
    fine = compute_quasi_interpolant_norm(
        family, problem, 2, normalise=normalise, refinement=4
    )

    # A supremum over nested subspaces increases under refinement.
    assert coarse <= fine
    assert fine == pytest.approx(expected, abs=5e-4)


def _record(**estimators) -> TrainingRecord:
    return TrainingRecord(
        loss=jnp.array([0.0, 0.0]),
        terms={"eta": jnp.array([4.0, 9.0])},
        energy_error=jnp.array([2.0, 4.0]),
        relative_loss=jnp.array([0.0, 0.0]),
        relative_energy_error=jnp.array([0.0, 0.0]),
        robustness=jnp.array([0.0, 0.0]),
        estimators=estimators,
    )


def test_bound_ratios_divide_both_sides_by_the_error() -> None:
    ratios = compute_bound_ratios(
        ErrorBound("eta", "rho", 1.5, 2.0), _record(rho=jnp.array([1.0, 4.0]))
    )

    assert np.allclose(ratios.lower, [1.0, 0.75])
    assert np.allclose(ratios.upper, [(1.5 * 2.0 + 2.0) / 2.0, (1.5 * 3.0 + 4.0) / 4.0])


def test_bound_ratios_need_both_terms() -> None:
    with pytest.raises(KeyError):
        compute_bound_ratios(ErrorBound("eta", "rho", 1.0, 1.0), _record())


def test_monitor_records_the_estimators() -> None:
    def create_monitor(**options):
        return create_training_monitor(
            lambda model: {"a": model**2},
            jnp.abs,
            jnp.asarray(2.0),
            **options,
        )

    with_estimators = create_monitor(estimators=lambda model: {"b": 3.0 * model})(
        jnp.asarray(2.0)
    )
    without = create_monitor()(jnp.asarray(2.0))

    assert float(with_estimators.estimators["b"]) == pytest.approx(6.0)
    assert without.estimators == {}
    assert float(without.robustness) == pytest.approx(1.0)
