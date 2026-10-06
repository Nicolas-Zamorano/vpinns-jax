"""
Discrete weak residuals of trial functions against test spaces.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax_fem.assembly import assemble_vector, integrate_cellwise
from jax_fem.forms import elliptic_bilinear_form, elliptic_linear_form
from jax_fem.function import FiniteElementFunctionEvaluation

from vpinns_jax.test_spaces.fe_space import FETestSpace
from vpinns_jax.test_spaces.patch_space import PatchTestSpace, assemble_patch_vector


def compute_fe_residual(
    fe_test_space: FETestSpace, trial: FiniteElementFunctionEvaluation
) -> jax.Array:
    """
    Weak residual of a trial function evaluated at the quadrature points.

    Parameters
    ----------
    fe_test_space : FETestSpace
        Test space.
    trial : FiniteElementFunctionEvaluation
        Trial values, shape ``(K, Q, 1, 1)``, and gradients, shape
        ``(K, Q, 1, d)``, at ``fe_test_space.basis.physical_points``, e.g.
        from ``evaluate_network``.

    Returns
    -------
    residual : jax.Array
        ``a(w, psi_i) - l(psi_i)`` for every DOF ``i``, shape
        ``(dofs_test, 1)``.

    Notes
    -----
    The trial evaluation enters the ``jax_fem`` elliptic forms as a trial
    basis with a single function, so the residual uses exactly the forms of
    the finite element discretization, integrated with the test space's
    quadrature.
    """
    basis = fe_test_space.basis
    coefficients = fe_test_space.coefficients
    pointwise = elliptic_bilinear_form(basis, trial, coefficients) - (
        elliptic_linear_form(basis, coefficients)
    )
    return assemble_vector(
        integrate_cellwise(pointwise, basis.physical_weights), basis.space
    )


def compute_interpolated_fe_residual(
    fe_test_space: FETestSpace, dof_values: jax.Array
) -> jax.Array:
    """
    Weak residual of a finite element trial function.

    Parameters
    ----------
    fe_test_space : FETestSpace
        Test space whose space also holds the trial function.
    dof_values : jax.Array
        Trial DOF values, shape ``(dofs_test, 1)``.

    Returns
    -------
    residual : jax.Array
        ``stiffness @ w - load``, shape ``(dofs_test, 1)``.
    """
    return fe_test_space.stiffness @ dof_values - fe_test_space.load


def compute_patch_residual(
    patch_test_space: PatchTestSpace, trial: FiniteElementFunctionEvaluation
) -> jax.Array:
    """
    Weak residual of a trial function against the patch test functions.

    Parameters
    ----------
    patch_test_space : PatchTestSpace
        Patch test space.
    trial : FiniteElementFunctionEvaluation
        Trial values, shape ``(K, Q, 1, 1)``, and gradients, shape
        ``(K, Q, 1, d)``, at ``patch_test_space.basis.physical_points``, e.g.
        from ``evaluate_network``.

    Returns
    -------
    residual : jax.Array
        ``a(w, psi_i) - l(psi_i)`` for every patch ``i``, shape
        ``(N_patch, 1)``.

    Notes
    -----
    Every integration cell is visited once, with the trial evaluated once at
    its quadrature points, and contributes to all patches listed on it. Only
    the entries of ``family.interior_patches`` are residuals of test
    functions in ``H^1_0``.
    """
    test_functions = patch_test_space.test_functions
    coefficients = patch_test_space.coefficients
    pointwise = elliptic_bilinear_form(test_functions, trial, coefficients) - (
        elliptic_linear_form(test_functions, coefficients)
    )
    return assemble_patch_vector(
        integrate_cellwise(pointwise, patch_test_space.basis.physical_weights),
        patch_test_space.family,
    )


def compute_patch_residual_terms(
    patch_test_space: PatchTestSpace,
    trial: FiniteElementFunctionEvaluation,
    flux_divergence: jax.Array,
) -> jax.Array:
    """
    Squared weighted strong residuals of a smooth trial function on every patch.

    Parameters
    ----------
    patch_test_space : PatchTestSpace
        Patch test space.
    trial : FiniteElementFunctionEvaluation
        Trial values, shape ``(K, Q, 1, 1)``, and gradients, shape
        ``(K, Q, 1, d)``, at ``patch_test_space.basis.physical_points``.
    flux_divergence : jax.Array
        ``div(A grad w)`` at the same points, shape ``(K, Q, 1, 1)``, e.g. from
        ``evaluate_network_flux_divergence``.

    Returns
    -------
    squared_terms : jax.Array
        ``eta_R,i^2 = h_i^2 ||phi_i^(1/2) rho(w)||^2_(P_i)`` for every patch
        ``i``, interior and boundary, shape ``(N_patch, 1)``.

    Notes
    -----
    The strong residual is ``rho(w) = f + div(A grad w) - beta . grad w - c w``
    and is weighted by the raw weight ``phi_i``. The terms bound the part of
    the residual that the patch test space cannot see. Edge residuals are
    omitted, which is exact for trial functions with continuous fluxes, such
    as networks with smooth activations and coefficients; the shape-regularity
    constants of the patches are not included.
    """
    coefficients = patch_test_space.coefficients
    strong_residual = (
        coefficients.source
        + flux_divergence
        - coefficients.advection @ trial.gradients.mT
        - coefficients.reaction * trial.values
    )
    local_terms = integrate_cellwise(
        patch_test_space.raw_weights.values * strong_residual**2,
        patch_test_space.basis.physical_weights,
    )
    diameters = jnp.expand_dims(patch_test_space.family.patch_diameters, axis=-1)
    return diameters**2 * assemble_patch_vector(local_terms, patch_test_space.family)
