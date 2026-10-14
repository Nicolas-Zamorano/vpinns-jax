"""Why the patch weights must be normalised: a fixed error under refinement.

On the unit square, the error ``e = u - w = sin(pi x) sin(pi y)`` is held fixed,
so ``|grad e| = pi / sqrt(2)`` for every grid, and the residual of ``w`` is
``r_i = (grad e, grad psi_i)``. For the l^inf cone families with spacing
``1/n``, the script computes, without any training,

- ``eta_P / |grad e|`` with the normalised weights ``psi_i = phi_i / S``, which
  settles at a fixed fraction of the error;
- the same with the raw weights ``phi_i``, which halves with every refinement,
  so a bound built on it cannot hold uniformly;
- ``eta_D / |grad e|``, the decoupled dual norm with the normalised weights,
  which also decays with ``h``;
- ``eta_R / |grad e|``, with ``eta_R,i = h_i c_2 ||phi_i^(1/2) lap e||`` and
  ``c_2 = 1 / pi``, the approximability term, which is ``O(h)``.

The ratios are plotted against ``n`` and saved under ``experiments/``.

Run with:

    uv run python examples/04_shepard_normalisation.py
"""

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax_fem.diagnostics import create_timestamped_output_directory, save_plot
from jax_fem.problem import (
    EllipticProblem,
    constant_scalar_coefficient,
    constant_tensor_coefficient,
    constant_vector_coefficient,
    create_full_boundary_dirichlet_condition,
)

from vpinns_jax.analysis import line_style, show_grid
from vpinns_jax.losses import (
    compute_patch_residual,
    compute_patch_residual_terms,
    diagonal_riesz_loss,
    restrict_to_free,
    riesz_loss,
)
from vpinns_jax.models import evaluate_network, evaluate_network_flux_divergence
from vpinns_jax.test_spaces import (
    PatchTestSpace,
    create_linf_grid_family,
    create_patch_test_space,
)


def error_function(x: jnp.ndarray) -> jnp.ndarray:
    """
    The fixed error ``e = sin(pi x) sin(pi y)``.

    Parameters
    ----------
    x : jnp.ndarray
        Point, shape ``(2,)``.

    Returns
    -------
    value : jnp.ndarray
        ``e(x)``, shape ``()``.
    """
    return jnp.sin(jnp.pi * x[0]) * jnp.sin(jnp.pi * x[1])


def _dual_norms(space: PatchTestSpace) -> tuple[float, float]:
    """
    Coupled and decoupled dual norms of the residual of the fixed error.

    Parameters
    ----------
    space : PatchTestSpace
        Patch test space of a problem with ``f = 0`` and ``A = I``.

    Returns
    -------
    coupled : float
        ``eta_P = (r^T P^{-1} r)^(1/2)``.
    decoupled : float
        ``eta_D = (r^T D^{-1} r)^(1/2)``.
    """
    interior = space.family.interior_patches
    trial = evaluate_network(error_function, space.basis.physical_points)
    residual = jnp.take(compute_patch_residual(space, trial), interior, axis=-2)
    gram = restrict_to_free(space.gram, interior)
    coupled = jnp.sqrt(riesz_loss(residual, jnp.linalg.cholesky(gram)))
    decoupled = jnp.sqrt(diagonal_riesz_loss(residual, jnp.diagonal(gram)))
    return float(coupled), float(decoupled)


def _residual_estimate(space: PatchTestSpace, problem: EllipticProblem) -> float:
    """
    Residual term of the fixed error with ``c_2 = 1 / pi``.

    Parameters
    ----------
    space : PatchTestSpace
        Patch test space.
    problem : EllipticProblem
        Problem with ``f = 0`` and ``A = I``.

    Returns
    -------
    residual_estimate : float
        ``(sum_i eta_R,i^2)^(1/2)``.
    """
    points = space.basis.physical_points
    trial = evaluate_network(error_function, points)
    flux_divergence = evaluate_network_flux_divergence(
        error_function, problem.diffusion, points
    )
    terms = compute_patch_residual_terms(space, trial, flux_divergence)
    return float(jnp.sqrt(jnp.sum(terms)) / jnp.pi)


def main(grid_sizes: tuple[int, ...] = (4, 8, 16, 32)) -> None:
    """
    Print and plot the estimator ratios for a fixed error.

    Parameters
    ----------
    grid_sizes : tuple[int, ...]
        Numbers of grid squares per side.
    """
    error_norm = float(jnp.pi / jnp.sqrt(2.0))
    labels = ("eta_P (psi)", "eta_P (phi)", "eta_D (psi)", "eta_R")
    rows = []
    print(
        "    n"
        + "".join(f"{label:>14s}" for label in labels)
        + "   (ratios to |grad e|)"
    )
    for cells_per_side in grid_sizes:
        family = create_linf_grid_family(
            np.ones((cells_per_side, cells_per_side), dtype=bool),
            spacing=1.0 / cells_per_side,
        )
        # With f = 0 and A = I, the residual of the trial e is (grad e, grad psi_i).
        problem = EllipticProblem(
            diffusion=constant_tensor_coefficient(jnp.eye(2)),
            advection=constant_vector_coefficient(jnp.zeros(2)),
            reaction=constant_scalar_coefficient(0.0),
            source=constant_scalar_coefficient(0.0),
            dirichlet_conditions=(
                create_full_boundary_dirichlet_condition(
                    family.mesh, constant_scalar_coefficient(0.0)
                ),
            ),
        )
        normalised = create_patch_test_space(family, problem)
        raw = create_patch_test_space(family, problem, normalise=False)
        coupled, decoupled = _dual_norms(normalised)
        raw_coupled, _ = _dual_norms(raw)
        ratios = (
            np.array(
                [
                    coupled,
                    raw_coupled,
                    decoupled,
                    _residual_estimate(normalised, problem),
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
    axis.set_xlabel("n (grid squares per side)")
    axis.set_ylabel(r"ratio to $\|\nabla e\|$")
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
