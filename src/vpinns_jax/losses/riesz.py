"""Robust Riesz loss ``L_Riesz(w_theta) = r^T G^{-1} r = ||z_h||_A^2`` (math reference Section 2.3).

Two evaluation paths:

- **Global Gram matrix** (``assemble_energy_gram_matrix`` +
  ``riesz_loss_global``), for an FE test space: ``G_ij = (psi_i, psi_j)_A``,
  assembled from the energy inner product form (diffusion + symmetrized
  reaction, *no* advection term -- ``(., .)_A`` is symmetric by
  construction, math reference Section 1), then inverted by a dense solve
  on the free-DOF submatrix.
- **Single-DOF patch collapse** (``patch_self_energy`` +
  ``riesz_loss_patch``), for a patch test space with exactly one basis
  function per patch: the local Gram "matrix" is the scalar
  ``(varphi_i, varphi_i)_A``, so ``L_Riesz`` collapses to
  ``sum_i r_i^2 / (varphi_i, varphi_i)_A`` -- the "single-function-per-patch
  identity" (math reference Section 2.3), with *no approximation gap*
  relative to the general Gram-matrix definition.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.experimental import sparse as jax_sparse
from jax_fem.assembly import GlobalMatrix, integrate_cellwise
from jax_fem.space import CellBasis

from vpinns_jax.pde.base import EllipticPDE, energy_reaction_coefficient
from vpinns_jax.test_spaces.fe_space import FETestSpace
from vpinns_jax.test_spaces.patch_space import PatchTestSpace, gather_patch_quadrature

__all__ = [
    "energy_bilinear_form",
    "assemble_energy_gram_matrix",
    "restrict_to_free_dofs",
    "riesz_loss_global",
    "patch_self_energy",
    "riesz_loss_patch",
]


def energy_bilinear_form(
    basis: CellBasis, diffusion: jax.Array, symmetrized_reaction: jax.Array
) -> jax.Array:
    r"""``(v, w)_A`` pointwise integrand: ``grad(v).mu.grad(w) + (sigma - div(beta)/2) v w``.

    No advection term -- unlike ``jax_fem.forms.elliptic_bilinear_form``,
    this is symmetric by construction. Returns shape ``(K, Q, N_phi, N_phi)``.
    """
    diffusion_term = basis.gradients @ diffusion @ basis.gradients.mT
    reaction_term = symmetrized_reaction * (basis.values @ basis.values.mT)
    return diffusion_term + reaction_term


def assemble_energy_gram_matrix(
    fe_test_space: FETestSpace, problem: EllipticPDE
) -> GlobalMatrix:
    """``G_ij = (psi_i, psi_j)_A`` over every global FE DOF pair, shape ``(N_dofs, N_dofs)``."""
    basis = fe_test_space.basis
    symmetrized_reaction = energy_reaction_coefficient(problem)(basis.physical_points)
    pointwise = energy_bilinear_form(
        basis, fe_test_space.coefficients.diffusion, symmetrized_reaction
    )
    local_matrices = integrate_cellwise(pointwise, basis.physical_weights)

    cells_to_dofs = fe_test_space.space.cells_to_dofs
    number_of_dofs = fe_test_space.space.number_of_dofs
    row_indices = jnp.broadcast_to(cells_to_dofs[:, :, None], local_matrices.shape)
    col_indices = jnp.broadcast_to(cells_to_dofs[:, None, :], local_matrices.shape)
    flat_indices = jnp.stack((row_indices.reshape(-1), col_indices.reshape(-1)), axis=-1)
    flat_data = local_matrices.reshape(-1)
    return jax_sparse.BCOO(
        (flat_data, flat_indices), shape=(number_of_dofs, number_of_dofs)
    )


def restrict_to_free_dofs(gram: GlobalMatrix, free_dofs: jax.Array) -> jax.Array:
    """Dense ``(N_free, N_free)`` restriction of a global sparse Gram matrix.

    A dense restriction (rather than a sparse condensation, cf.
    ``jax_fem.constraints.condense_dirichlet_system``) is a deliberate
    simplicity choice at this library's scale; a sparse solve would be
    needed for a large free-DOF count.
    """
    return gram.todense()[free_dofs][:, free_dofs]


def riesz_loss_global(residual_free: jax.Array, gram_free_dense: jax.Array) -> jax.Array:
    """``L_Riesz = r^T G^{-1} r`` via a dense solve on the free-DOF Gram submatrix.

    Parameters
    ----------
    residual_free : jax.Array
        ``r`` restricted to free (interior) DOFs, shape ``(N_free,)``.
    gram_free_dense : jax.Array
        ``restrict_to_free_dofs(assemble_energy_gram_matrix(...), free_dofs)``,
        shape ``(N_free, N_free)``.
    """
    representative = jnp.linalg.solve(gram_free_dense, residual_free)
    return jnp.sum(residual_free * representative)


def patch_self_energy(patch_space: PatchTestSpace, problem: EllipticPDE) -> jax.Array:
    """``(varphi_i, varphi_i)_A`` for every patch ``i``, shape ``(N_patch,)``.

    Trial-independent: a fixed, positive per-patch constant determined by
    the PDE data and the patch cover alone (math reference Section 2.3),
    computed once and reused every training step.
    """
    patch_quadrature = gather_patch_quadrature(patch_space)
    diffusion = problem.diffusion(patch_quadrature.physical_points)
    symmetrized_reaction = energy_reaction_coefficient(problem)(
        patch_quadrature.physical_points
    )
    diffusion_term = (
        patch_quadrature.hat_gradients
        @ diffusion
        @ jnp.swapaxes(patch_quadrature.hat_gradients, -1, -2)
    )
    reaction_term = (
        symmetrized_reaction * patch_quadrature.hat_values * patch_quadrature.hat_values
    )
    pointwise = diffusion_term + reaction_term
    weighted = patch_quadrature.physical_weights * pointwise
    number_of_patches = patch_quadrature.physical_points.shape[0]
    return jnp.sum(weighted.reshape(number_of_patches, -1), axis=-1)


def riesz_loss_patch(residual_patch: jax.Array, self_energy: jax.Array) -> jax.Array:
    """``L_Riesz = sum_i r_i^2 / (varphi_i, varphi_i)_A`` (single-DOF-per-patch collapse)."""
    return jnp.sum(residual_patch**2 / self_energy)
