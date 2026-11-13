"""Why the patch weights must be normalised: a fixed error under refinement.

The problem is the advection-diffusion-reaction problem
``-div(A grad u) + beta . grad u + c u = f`` of
``create_simple_elliptic_problem`` on the unit square, with nonhomogeneous
Dirichlet data ``g``. The trial function ``w = g_bar``, the transfinite
interpolation of ``g`` into the square, which is the ansatz
``w = B u_theta + g_bar`` with ``u_theta = 0``, is held fixed. Its error
``e = u - w`` lies in ``H^1_0`` and is the same for every grid, so its energy
norm ``|e|_A`` is computed once, on a fine mesh, and the residual of ``w`` is
``r_i = a(w, psi_i) - l(psi_i) = -a(e, psi_i)``. For the l^inf cone families
with spacing ``1/n``, the script computes, without any training,

- ``eta_P / |e|_A`` with the normalised weights ``psi_i = phi_i / S``, which
  settles at a fixed fraction of the error once the grid resolves it;
- the same with the raw weights ``phi_i``, which then halves with every
  refinement, so a bound built on it cannot hold uniformly;
- ``eta_D / |e|_A``, the decoupled dual norm with the normalised weights,
  which also decays with ``h``;
- ``eta_R / |e|_A``, with ``eta_R,i = h_i c_2 ||phi_i^(1/2) rho(w)||``, the
  strong residual ``rho(w) = f + div(A grad w) - beta . grad w - c w``, and
  ``c_2 = 1 / pi``, the approximability term, which is ``O(h)``.

The ratios are plotted against ``n`` and saved under ``experiments/``.

Run with:

    uv run python examples/04_shepard_normalisation.py
"""

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax_fem.diagnostics import (
    create_timestamped_output_directory,
    energy_error_functional,
    save_plot,
)
from jax_fem.element import LagrangeTriangleP1
from jax_fem.mesh import create_gmsh_unit_square_mesh
from jax_fem.problem import EllipticProblem, create_simple_elliptic_problem
from jax_fem.reference_cell import ReferenceTriangle
from jax_fem.space import create_cell_basis, create_finite_element_space
from matplotlib.ticker import NullFormatter

from vpinns_jax.analysis import compute_network_error, line_style, show_grid
from vpinns_jax.losses import (
    compute_patch_residual,
    compute_patch_residual_terms,
    diagonal_riesz_loss,
    restrict_to_free,
    riesz_loss,
)
from vpinns_jax.models import (
    Network,
    TransfiniteLift,
    evaluate_network,
    evaluate_network_flux_divergence,
)
from vpinns_jax.test_spaces import (
    PatchTestSpace,
    create_linf_grid_family,
    create_patch_test_space,
)


def _dual_norms(space: PatchTestSpace, trial_function: Network) -> tuple[float, float]:
    """
    Coupled and decoupled dual norms of the residual of a fixed trial function.

    Parameters
    ----------
    space : PatchTestSpace
        Patch test space.
    trial_function : Network
        The fixed trial function ``w``.

    Returns
    -------
    coupled : float
        ``eta_P = (r^T P^{-1} r)^(1/2)``.
    decoupled : float
        ``eta_D = (r^T D^{-1} r)^(1/2)``.
    """
    interior = space.family.interior_patches
    trial = evaluate_network(trial_function, space.basis.physical_points)
    residual = jnp.take(compute_patch_residual(space, trial), interior, axis=-2)
    gram = restrict_to_free(space.gram, interior)
    coupled = jnp.sqrt(riesz_loss(residual, jnp.linalg.cholesky(gram)))
    decoupled = jnp.sqrt(diagonal_riesz_loss(residual, jnp.diagonal(gram)))
    return float(coupled), float(decoupled)


def _residual_estimate(
    space: PatchTestSpace, problem: EllipticProblem, trial_function: Network
) -> float:
    """
    Residual term of a fixed trial function with ``c_2 = 1 / pi``.

    Parameters
    ----------
    space : PatchTestSpace
        Patch test space.
    problem : EllipticProblem
        Problem providing the diffusion of the strong residual.
    trial_function : Network
        The fixed trial function ``w``.

    Returns
    -------
    residual_estimate : float
        ``(sum_i eta_R,i^2)^(1/2)``.
    """
    points = space.basis.physical_points
    trial = evaluate_network(trial_function, points)
    flux_divergence = evaluate_network_flux_divergence(
        trial_function, problem.diffusion, points
    )
    terms = compute_patch_residual_terms(space, trial, flux_divergence)
    return float(jnp.sqrt(jnp.sum(terms)) / jnp.pi)


def main(
    grid_sizes: tuple[int, ...] = (4, 8, 16, 32, 64),
    error_cells_per_side: int = 64,
) -> None:
    """
    Print and plot the estimator ratios for a fixed error.

    Parameters
    ----------
    grid_sizes : tuple[int, ...]
        Numbers of grid squares per side.
    error_cells_per_side : int
        Resolution of the mesh on which ``|e|_A`` is measured.
    """
    element = LagrangeTriangleP1(reference_cell=ReferenceTriangle())
    error_mesh = create_gmsh_unit_square_mesh(error_cells_per_side)
    error_basis = create_cell_basis(
        create_finite_element_space(error_mesh, element),
        element.reference_cell.create_quadrature(5),
    )
    # The mesh only tags the boundary of the problem's Dirichlet condition; the
    # patch spaces use the coefficient functions and the lift the data.
    problem = create_simple_elliptic_problem(error_mesh)
    (dirichlet_condition,) = problem.dirichlet_conditions
    lift = TransfiniteLift(dirichlet_condition.value)
    error_norm = float(
        compute_network_error(lift, error_basis, problem, energy_error_functional)
    )

    labels = ("eta_P (psi)", "eta_P (phi)", "eta_D (psi)", "eta_R")
    rows = []
    print(f"Fixed trial w = g_bar: |e|_A = |u - w|_A = {error_norm:.4f}")
    print(
        "    n" + "".join(f"{label:>14s}" for label in labels) + "   (ratios to |e|_A)"
    )
    for cells_per_side in grid_sizes:
        family = create_linf_grid_family(
            np.ones((cells_per_side, cells_per_side), dtype=bool),
            spacing=1.0 / cells_per_side,
        )
        normalised = create_patch_test_space(family, problem)
        raw = create_patch_test_space(family, problem, normalise=False)
        coupled, decoupled = _dual_norms(normalised, lift)
        raw_coupled, _ = _dual_norms(raw, lift)
        ratios = (
            np.array(
                [
                    coupled,
                    raw_coupled,
                    decoupled,
                    _residual_estimate(normalised, problem, lift),
                ]
            )
            / error_norm
        )
        rows.append(ratios)
        print(f"{cells_per_side:5d}" + "".join(f"{ratio:14.3f}" for ratio in ratios))

    table = np.stack(rows)
    figure, axis = plt.subplots(figsize=(6, 4))
    legend_labels = (
        r"$\eta_P$ with $\psi_i$",
        r"$\eta_P$ with $\varphi_i$",
        r"$\eta_D$ with $\psi_i$",
        r"$\eta_R$",
    )
    for column, label in enumerate(legend_labels):
        axis.loglog(
            grid_sizes, table[:, column], marker="o", label=label, **line_style(column)
        )
    axis.set_xticks(grid_sizes, labels=[str(n) for n in grid_sizes])
    axis.xaxis.set_minor_formatter(NullFormatter())
    axis.set_xlabel("n (grid squares per side)")
    axis.set_ylabel(r"ratio to $\|e\|_A$")
    axis.set_title("Fixed error under refinement")
    axis.legend()
    show_grid(axis)

    output_directory = create_timestamped_output_directory(
        "experiments", "04_shepard_normalisation"
    )
    save_plot(figure, output_directory / "ratios.png")
    print(f"Saved plots to {output_directory}/")


if __name__ == "__main__":
    main()
