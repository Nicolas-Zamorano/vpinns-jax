"""Tests for the Shepard-normalised patch test space."""

import jax.numpy as jnp
import numpy as np
import pytest
from jax_fem.problem import (
    create_poisson_sin_sin_problem,
    create_simple_elliptic_problem,
)

from vpinns_jax.losses import (
    compute_fe_residual,
    compute_patch_residual,
    restrict_to_free,
)
from vpinns_jax.models import MultiLayerPerceptron, evaluate_network
from vpinns_jax.test_spaces import (
    PatchFunctions,
    PatchTestSpace,
    create_linf_grid_family,
    create_patch_test_space,
    create_vertex_star_family,
)


@pytest.fixture(scope="module")
def linf_family():
    return create_linf_grid_family(np.ones((4, 4), dtype=bool), spacing=0.25)


def test_normalised_weights_are_a_partition_of_unity(linf_family) -> None:
    space = create_patch_test_space(
        linf_family, create_poisson_sin_sin_problem(linf_family.mesh)
    )
    values = space.test_functions.values

    assert jnp.allclose(jnp.sum(values, axis=-2), 1.0, atol=1e-15)
    assert jnp.allclose(
        jnp.sum(space.test_functions.gradients, axis=-2), 0.0, atol=1e-13
    )
    sums = jnp.sum(space.raw_weights.values, axis=-2, keepdims=True)

    assert jnp.all((values >= 0.0) & (values <= 1.0 + 1e-15))
    assert jnp.allclose(values * sums, space.raw_weights.values)


def test_raw_weights_are_kept_without_normalisation(linf_family) -> None:
    space = create_patch_test_space(
        linf_family, create_poisson_sin_sin_problem(linf_family.mesh), normalise=False
    )
    sums = jnp.sum(space.raw_weights.values, axis=-2)
    lower, upper = linf_family.covering_bounds

    assert space.test_functions is space.raw_weights
    assert jnp.all((sums >= lower) & (sums <= upper))
    assert jnp.max(sums) > 1.5


@pytest.mark.parametrize(
    ("problem_name", "fe_name"),
    [("sin_sin_problem", "poisson_fe"), ("advection_problem", "advection_fe")],
)
def test_vertex_star_space_reduces_to_the_fe_test_space(
    problem_name, fe_name, unit_square_mesh, prng_key, request
) -> None:
    fe = request.getfixturevalue(fe_name)
    family = create_vertex_star_family(unit_square_mesh)
    space = create_patch_test_space(
        family, request.getfixturevalue(problem_name), quadrature_degree=4
    )
    network = MultiLayerPerceptron(width=8, depth=2, key=prng_key)

    assert jnp.allclose(space.test_functions.values, fe.basis.values)
    assert jnp.allclose(
        restrict_to_free(space.gram, family.interior_patches),
        restrict_to_free(fe.gram, fe.free_dofs),
    )
    assert jnp.allclose(
        compute_patch_residual(
            space, evaluate_network(network, space.basis.physical_points)
        ),
        compute_fe_residual(fe, evaluate_network(network, fe.basis.physical_points)),
    )


def test_gram_matrix_is_symmetric_positive_definite_on_interior_patches(
    linf_family,
) -> None:
    space = create_patch_test_space(
        linf_family, create_simple_elliptic_problem(linf_family.mesh)
    )
    gram = restrict_to_free(space.gram, linf_family.interior_patches)

    assert jnp.allclose(gram, gram.T, atol=1e-14)
    assert jnp.all(jnp.linalg.eigvalsh(gram) > 0.0)


def test_mismatched_patch_functions_are_rejected(linf_family) -> None:
    space = create_patch_test_space(
        linf_family, create_poisson_sin_sin_problem(linf_family.mesh)
    )
    with pytest.raises(ValueError, match="test_functions"):
        PatchTestSpace(
            family=space.family,
            basis=space.basis,
            coefficients=space.coefficients,
            test_functions=PatchFunctions(
                values=space.test_functions.values[..., :2, :],
                gradients=space.test_functions.gradients[..., :2, :],
            ),
            raw_weights=space.raw_weights,
            gram=space.gram,
        )
