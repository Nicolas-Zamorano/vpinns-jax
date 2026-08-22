"""tests/test_patch_space.py: patch cover bound, geometry, and sub-triangulation."""

from __future__ import annotations

import jax.numpy as jnp

from vpinns_jax.test_spaces import create_patch_test_space, gather_patch_quadrature


def test_cover_constants_are_two_sided_and_positive(unit_square_mesh):
    patches = create_patch_test_space(unit_square_mesh)
    assert 0.0 < patches.cover_constant_lower <= patches.cover_constant_upper


def test_patch_diameters_are_positive_and_bounded_by_domain_size(unit_square_mesh):
    patches = create_patch_test_space(unit_square_mesh)
    assert jnp.all(patches.patch_diameter > 0.0)
    # unit square has diameter sqrt(2); no patch can exceed the whole domain.
    assert jnp.all(patches.patch_diameter <= jnp.sqrt(2.0) + 1e-9)


def test_every_patch_has_at_least_three_incident_cells(unit_square_mesh):
    patches = create_patch_test_space(unit_square_mesh)
    valence = jnp.sum(patches.patch_mask, axis=-1)
    assert jnp.all(valence >= 3)


def test_patch_centers_are_interior_vertices_only(unit_square_mesh):
    patches = create_patch_test_space(unit_square_mesh)
    coordinates = patches.space.dof_coordinates[patches.interior_vertices, 0, :]
    on_boundary = (
        jnp.isclose(coordinates[:, 0], 0.0)
        | jnp.isclose(coordinates[:, 0], 1.0)
        | jnp.isclose(coordinates[:, 1], 0.0)
        | jnp.isclose(coordinates[:, 1], 1.0)
    )
    assert not jnp.any(on_boundary)


def test_patch_hat_function_is_nonnegative_and_bounded_by_one(unit_square_mesh):
    patches = create_patch_test_space(unit_square_mesh)
    quadrature = gather_patch_quadrature(patches)
    values = quadrature.hat_values[quadrature.mask]
    assert jnp.all(values >= -1e-12)
    assert jnp.all(values <= 1.0 + 1e-12)


def test_reusing_an_existing_fe_space_shares_the_instance(unit_square_mesh, sin_sin_problem):
    from vpinns_jax.test_spaces import create_fe_test_space

    fe = create_fe_test_space(unit_square_mesh, sin_sin_problem)
    patches = create_patch_test_space(unit_square_mesh, space=fe.space)
    assert patches.space is fe.space
