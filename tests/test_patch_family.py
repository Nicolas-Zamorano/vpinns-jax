"""Tests for the patch families and their validation."""

import jax.numpy as jnp
import numpy as np
import pytest
from jax_fem.assembly import integrate_cellwise
from jax_fem.problem import create_poisson_sin_sin_problem

from vpinns_jax.test_spaces import (
    PatchFamily,
    assemble_patch_vector,
    create_linf_grid_family,
    create_patch_test_space,
    create_vertex_star_family,
)
from vpinns_jax.test_spaces.patch_family import _create_patch_family


def _centres(family: PatchFamily) -> np.ndarray:
    """Grid points come first in the integration mesh of an l^inf family."""
    coordinates = np.asarray(family.mesh.vertex_coordinates)[..., 0, :]
    return coordinates[: family.number_of_patches]


def _weight_integrals(family: PatchFamily) -> np.ndarray:
    space = create_patch_test_space(
        family, create_poisson_sin_sin_problem(family.mesh), quadrature_degree=2
    )
    integrals = assemble_patch_vector(
        integrate_cellwise(space.raw_weights.values, space.basis.physical_weights),
        family,
    )
    return np.asarray(integrals)[:, 0]


@pytest.mark.parametrize("cells_per_side", [1, 3, 4])
def test_linf_family_on_the_unit_square(cells_per_side) -> None:
    spacing = 1.0 / cells_per_side
    family = create_linf_grid_family(
        np.ones((cells_per_side, cells_per_side), dtype=bool), spacing=spacing
    )
    centres = _centres(family)
    on_boundary = np.sum(np.isclose(centres, 0.0) | np.isclose(centres, 1.0), axis=-1)

    assert family.number_of_patches == (cells_per_side + 1) ** 2
    assert family.covering_bounds == (1.0, 2.0)
    assert np.array_equal(np.asarray(family.boundary_patch_mask), on_boundary > 0)
    assert family.interior_patches.shape == ((cells_per_side - 1) ** 2,)
    # Corner, mid-side, and interior configurations.
    expected_diameters = np.array([2.0 * np.sqrt(2.0), np.sqrt(5.0), np.sqrt(2.0)])
    expected_integrals = np.array([4.0, 2.0, 1.0]) / 3.0
    assert np.allclose(
        family.patch_diameters, spacing * expected_diameters[on_boundary]
    )
    assert np.allclose(
        _weight_integrals(family), spacing**2 * expected_integrals[on_boundary]
    )


def test_linf_family_on_the_l_shape() -> None:
    cells_per_unit = 2
    spacing = 1.0 / cells_per_unit
    cell_mask = np.ones((2 * cells_per_unit, 2 * cells_per_unit), dtype=bool)
    cell_mask[:cells_per_unit, cells_per_unit:] = False
    family = create_linf_grid_family(cell_mask, spacing=spacing, origin=(-1.0, -1.0))
    re_entrant = np.flatnonzero(np.all(np.isclose(_centres(family), 0.0), axis=-1))

    assert family.number_of_patches == (2 * cells_per_unit + 1) ** 2 - cells_per_unit**2
    assert family.covering_bounds == (1.0, 2.0)
    assert bool(family.boundary_patch_mask[re_entrant[0]])
    assert np.allclose(family.patch_diameters[re_entrant], 2.0 * np.sqrt(2.0) * spacing)
    assert np.allclose(_weight_integrals(family)[re_entrant], spacing**2)


def test_vertex_star_family_is_the_nodal_partition_of_unity(
    unit_square_mesh, poisson_fe
) -> None:
    family = create_vertex_star_family(unit_square_mesh)

    assert family.number_of_patches == unit_square_mesh.vertex_coordinates.shape[0]
    assert family.covering_bounds == (1.0, 1.0)
    assert jnp.array_equal(family.interior_patches, poisson_fe.free_dofs)
    assert jnp.allclose(family.patch_diameters, unit_square_mesh.vertex_patch_diameters)


def _vertex_star_arrays(mesh):
    number_of_cells = mesh.cells_to_vertices.shape[0]
    return (
        np.array(mesh.cells_to_vertices),
        np.ones((number_of_cells, 3), dtype=bool),
        np.array(np.broadcast_to(np.eye(3), (number_of_cells, 3, 3))),
    )


def test_family_without_boundary_patches_is_rejected(
    unit_square_mesh, poisson_fe
) -> None:
    cells_to_patches, cell_patch_mask, local_weights = _vertex_star_arrays(
        unit_square_mesh
    )
    is_boundary = np.isin(cells_to_patches, np.asarray(poisson_fe.dirichlet_dofs))
    renumbered = np.searchsorted(np.asarray(poisson_fe.free_dofs), cells_to_patches)
    with pytest.raises(ValueError, match="does not cover"):
        _create_patch_family(
            unit_square_mesh,
            renumbered,
            cell_patch_mask & ~is_boundary,
            local_weights,
            poisson_fe.free_dofs.shape[0],
        )


def test_discontinuous_weights_are_rejected(unit_square_mesh) -> None:
    cells_to_patches, cell_patch_mask, local_weights = _vertex_star_arrays(
        unit_square_mesh
    )
    local_weights[0, 0, 0] = 0.5
    with pytest.raises(ValueError, match="continuous"):
        _create_patch_family(
            unit_square_mesh,
            cells_to_patches,
            cell_patch_mask,
            local_weights,
            unit_square_mesh.vertex_coordinates.shape[0],
        )


def test_incomplete_cell_lists_are_rejected(unit_square_mesh) -> None:
    cells_to_patches, cell_patch_mask, local_weights = _vertex_star_arrays(
        unit_square_mesh
    )
    cell_patch_mask[0, 0] = False
    with pytest.raises(ValueError, match="must list"):
        _create_patch_family(
            unit_square_mesh,
            cells_to_patches,
            cell_patch_mask,
            local_weights,
            unit_square_mesh.vertex_coordinates.shape[0],
        )
