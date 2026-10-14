"""Tests for the finite element test space."""

import jax.numpy as jnp
import numpy as np
import pytest
from jax_fem.assembly import assemble_matrix, assemble_vector, integrate_cellwise
from jax_fem.forms import (
    elliptic_bilinear_form,
    elliptic_energy_bilinear_form,
    elliptic_linear_form,
)
from jax_fem.space import find_boundary_dofs

from vpinns_jax.test_spaces import FETestSpace


def _assemble_matrix(form, fe: FETestSpace) -> jnp.ndarray:
    basis = fe.basis
    local_matrices = integrate_cellwise(
        form(basis, basis, fe.coefficients), basis.physical_weights
    )
    return assemble_matrix(local_matrices, basis.space, basis.space).todense()


@pytest.mark.parametrize("fe_name", ["poisson_fe", "advection_fe"])
def test_operators_match_three_tier_assembly(fe_name, request) -> None:
    fe = request.getfixturevalue(fe_name)
    basis = fe.basis
    load = assemble_vector(
        integrate_cellwise(
            elliptic_linear_form(basis, fe.coefficients), basis.physical_weights
        ),
        basis.space,
    )

    assert jnp.allclose(
        fe.stiffness.todense(), _assemble_matrix(elliptic_bilinear_form, fe)
    )
    assert jnp.allclose(
        fe.gram.todense(), _assemble_matrix(elliptic_energy_bilinear_form, fe)
    )
    assert jnp.allclose(fe.load, load)


def test_free_and_dirichlet_dofs_partition_the_dofs(poisson_fe) -> None:
    space = poisson_fe.basis.space
    boundary_dofs = find_boundary_dofs(
        space, tuple(space.mesh.boundary_tag_names.values())
    )

    assert jnp.array_equal(poisson_fe.dirichlet_dofs, boundary_dofs)
    assert jnp.array_equal(
        poisson_fe.free_dofs,
        np.setdiff1d(np.arange(space.number_of_dofs), np.asarray(boundary_dofs)),
    )
    assert jnp.allclose(poisson_fe.dirichlet_values, 0.0)


def test_poisson_gram_matrix_equals_stiffness(poisson_fe) -> None:
    assert jnp.allclose(poisson_fe.gram.todense(), poisson_fe.stiffness.todense())


def test_inconsistent_dof_partition_is_rejected(poisson_fe) -> None:
    with pytest.raises(ValueError, match="partition"):
        FETestSpace(
            basis=poisson_fe.basis,
            coefficients=poisson_fe.coefficients,
            dirichlet_dofs=poisson_fe.dirichlet_dofs,
            dirichlet_values=poisson_fe.dirichlet_values,
            free_dofs=poisson_fe.free_dofs[1:],
            stiffness=poisson_fe.stiffness,
            load=poisson_fe.load,
            gram=poisson_fe.gram,
        )
