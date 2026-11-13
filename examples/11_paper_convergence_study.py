"""Loss functionals of the MF-VPINN under patch refinement.

Example 10 repeated on families with an increasing number of patches. The
problem is ``-Lap u = f`` in the unit square with ``u = sin(pi x) sin(pi y)``,
from ``create_poisson_sin_sin_problem``. The trial function is
``u_theta = zeta N_theta + g_bar``, with ``zeta = x (1 - x) y (1 - y)`` and the
Wachspress lift ``g_bar`` of the data, here zero.

For every number of cells per side ``n`` of ``grid_sizes``, the spacing is
``s = 1 / n`` and the families are the ``l^inf`` family of the paper and the
criss-cross family. The ``l^inf`` family has a square patch at every interior
grid point and the side and corner patches at the boundary grid points. On
every level, one network per loss of ``losses`` is trained with Adam, all from
the same initial weights. The losses are those of example 10:

======================  =============================  ===========
run                     loss                           family
======================  =============================  ===========
``mf``                  ``L_MF``                       ``l^inf``
``mf_rho``              ``L_MF + rho^2``               ``l^inf``
``eta_phi``             ``eta_phi^2``                  ``l^inf``
``eta_phi_rho``         ``eta_phi^2 + rho^2``          ``l^inf``
``eta_psi``             ``eta_psi^2``                  ``l^inf``
``eta_psi_rho``         ``eta_psi^2 + rho^2``          ``l^inf``
``eta_crisscross``      ``eta_P^2``                    criss-cross
``eta_crisscross_rho``  ``eta_P^2 + rho_P^2``          criss-cross
======================  =============================  ===========

By default the four losses with the estimator are trained. ``rho`` sums over
all patches of the ``l^inf`` family and ``rho_P`` over those of the
criss-cross family, boundary patches included. The efficiency index of a loss
is ``sqrt(L) / ||grad(u - u_theta)||_0`` at the end of its training.

For a fixed trial function with strong residual ``R = f + Lap u_theta``,
``sqrt(L_MF)``, ``rho``, and ``rho_P`` are ``O(s) ||R||_0``. For example,
``|r_i|^2 / |P_i| <= ||R||_{0,P_i}^2 / 6`` gives
``L_MF <= 2 ||R||_0^2 / (3 N)`` with ``N = (n - 1)^2`` test functions, and
``c_2(P_i) h_i = 2 sqrt(2) s / pi`` with ``S <= 2`` gives
``rho^2 <= (16 s^2 / pi^2) ||R||_0^2``. The main theorem keeps ``eta_psi``
between ``(||grad(u - u_theta)||_0 - (sqrt(S_max) / S_min) rho) / ||Pi_psi||``
and the error, and the criss-cross weights also sum to one. The raw ``l^inf``
weights sum to ``S``, which oscillates between 1 and 2 on the scale ``s``, so
their span does not approximate smooth functions in ``H^1``, and ``eta_phi`` of
a fixed smooth error also decays with ``s``.

For every level, the script plots the training quantities, the loss terms,
the robustness, and the solutions of every run, as example 10 does. It
evaluates every estimator on every trained network, prints the table and saves
it as ``n<n>/summary.csv``, and plots the relative error and the efficiency
index of all runs during training in ``n<n>/comparison.png``. After every
level it updates ``summary.csv`` with the rows of all levels, the table
``efficiency_index.csv`` of the index of every loss on every level, and the
plot of the errors and indices against ``n`` in ``convergence.png``. All files
are saved under ``experiments/``. The errors are measured on a finite element
mesh of the unit square, independent of the patches.

Run with:

    uv run python examples/11_paper_convergence_study.py
"""

import csv
import functools
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np
from jax_fem.diagnostics import (
    create_timestamped_output_directory,
    energy_error_functional,
    l2_error_functional,
    save_plot,
)
from jax_fem.element import LagrangeTriangleP1
from jax_fem.problem import (
    EllipticProblem,
    create_simple_elliptic_problem,
)
from jax_fem.reference_cell import ReferenceTriangle
from jax_fem.space import create_cell_basis, create_finite_element_space
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import NullLocator

from vpinns_jax.analysis import (
    compute_network_error,
    compute_solution_norm,
    create_training_monitor,
    line_style,
    plot_loss_terms,
    plot_network_solution,
    plot_robustness,
    plot_training_quantities,
    show_grid,
)
from vpinns_jax.losses import (
    compute_meshfree_patch_residual,
    compute_meshfree_patch_residual_terms,
    mf_vpinn_loss,
    restrict_to_free,
    riesz_loss,
)
from vpinns_jax.models import (
    ConstrainedAnsatz,
    ExactDistanceFunction,
    MultiLayerPerceptron,
    WachspressLift,
    evaluate_network,
    evaluate_network_flux_divergence,
)
from vpinns_jax.solvers import TrainingHistory, train_adam
from vpinns_jax.test_spaces import (
    MeshfreePatchTestSpace,
    create_crisscross_family,
    create_crisscross_unit_square_mesh,
    create_linf_vertex_family,
    create_meshfree_patch_test_space,
)

# Loss terms, named as in the paper; the names label the plots.
MF = r"$\mathcal{L}_{\mathrm{MF}}$"
ETA_PHI = r"$\eta_\varphi^2$"
ETA_PSI = r"$\eta_\psi^2$"
RHO = r"$\rho^2$"
ETA_CRISSCROSS = r"$\eta_{\mathcal{P}}^2$"
RHO_CRISSCROSS = r"$\rho_{\mathcal{P}}^2$"
# Terms of each family; rho sums over the patches of its family.
FAMILY_TERMS = {
    "linf": (MF, ETA_PHI, ETA_PSI, RHO),
    "crisscross": (ETA_CRISSCROSS, RHO_CRISSCROSS),
}
ALL_TERMS = FAMILY_TERMS["linf"] + FAMILY_TERMS["crisscross"]
# Columns of the square roots of the terms in the tables, and their headers.
COLUMNS = {
    MF: ("sqrt_L_MF", "sqrt(L_MF)"),
    ETA_PHI: ("eta_phi", "eta_phi"),
    ETA_PSI: ("eta_psi", "eta_psi"),
    RHO: ("rho", "rho"),
    ETA_CRISSCROSS: ("eta_crisscross", "eta_P"),
    RHO_CRISSCROSS: ("rho_crisscross", "rho_P"),
}
_ERROR = r"\|\nabla(u - u_\theta)\|_0"


class Loss(NamedTuple):
    """
    A loss of the study.

    Attributes
    ----------
    family : str
        ``"linf"`` or ``"crisscross"``.
    terms : tuple[str, ...]
        Names of the terms whose sum is the loss.
    """

    family: str
    terms: tuple[str, ...]


LOSSES = {
    "mf": Loss("linf", (MF,)),
    "mf_rho": Loss("linf", (MF, RHO)),
    "eta_phi": Loss("linf", (ETA_PHI,)),
    "eta_phi_rho": Loss("linf", (ETA_PHI, RHO)),
    "eta_psi": Loss("linf", (ETA_PSI,)),
    "eta_psi_rho": Loss("linf", (ETA_PSI, RHO)),
    "eta_crisscross": Loss("crisscross", (ETA_CRISSCROSS,)),
    "eta_crisscross_rho": Loss("crisscross", (ETA_CRISSCROSS, RHO_CRISSCROSS)),
}
DEFAULT_LOSSES = ("mf_rho", "eta_phi_rho", "eta_psi_rho", "eta_crisscross_rho")


class Level(NamedTuple):
    """
    The families of one spacing and what their loss terms need.

    Attributes
    ----------
    cells_per_side : int
        Number of grid cells per side ``n``.
    raw_space : MeshfreePatchTestSpace
        Space of the raw ``l^inf`` weights ``phi_i``.
    normalised_space : MeshfreePatchTestSpace
        Space of ``psi_i = phi_i / S`` on the same quadrature points.
    interior_areas : jax.Array
        ``|P_i|`` of the interior ``l^inf`` patches, shape ``(N,)``.
    raw_factor : jax.Array
        Lower Cholesky factor of the energy Gram matrix of the ``phi_i`` of
        the interior patches, shape ``(N, N)``.
    normalised_factor : jax.Array
        Lower Cholesky factor of the energy Gram matrix of the ``psi_i`` of
        the interior patches, shape ``(N, N)``.
    crisscross_space : MeshfreePatchTestSpace
        Space of the criss-cross family of the same spacing.
    crisscross_factor : jax.Array
        Lower Cholesky factor of the energy Gram matrix of its interior
        patches, shape ``(N_P, N_P)``.
    """

    cells_per_side: int
    raw_space: MeshfreePatchTestSpace
    normalised_space: MeshfreePatchTestSpace
    interior_areas: jax.Array
    raw_factor: jax.Array
    normalised_factor: jax.Array
    crisscross_space: MeshfreePatchTestSpace
    crisscross_factor: jax.Array


def create_level(
    cells_per_side: int, problem: EllipticProblem, quadrature_degree: int
) -> Level:
    """
    Spaces of the ``l^inf`` and criss-cross families of one spacing.

    Parameters
    ----------
    cells_per_side : int
        Number of grid cells per side ``n``.
    problem : EllipticProblem
        Problem whose source and coefficients the spaces integrate.
    quadrature_degree : int
        Exactness degree of the rule on the local triangles of every patch.

    Returns
    -------
    level : Level
        The spaces and the data of the loss terms.
    """
    family = create_linf_vertex_family(cells_per_side)
    raw_space = create_meshfree_patch_test_space(
        family, problem, quadrature_degree=quadrature_degree
    )
    normalised_space = create_meshfree_patch_test_space(
        family, problem, quadrature_degree=quadrature_degree, normalise=True
    )
    crisscross_space = create_meshfree_patch_test_space(
        create_crisscross_family(cells_per_side),
        problem,
        quadrature_degree=quadrature_degree,
    )
    interior = family.interior_patches
    return Level(
        cells_per_side=cells_per_side,
        raw_space=raw_space,
        normalised_space=normalised_space,
        interior_areas=jnp.take(family.patch_areas, interior),
        raw_factor=jnp.linalg.cholesky(restrict_to_free(raw_space.gram, interior)),
        normalised_factor=jnp.linalg.cholesky(
            restrict_to_free(normalised_space.gram, interior)
        ),
        crisscross_space=crisscross_space,
        crisscross_factor=jnp.linalg.cholesky(
            restrict_to_free(
                crisscross_space.gram, crisscross_space.family.interior_patches
            )
        ),
    )


def compute_loss_terms(
    level: Level,
    trial_function: ConstrainedAnsatz,
    problem: EllipticProblem,
    names: Sequence[str],
) -> dict[str, jax.Array]:
    """
    Squared estimators of a trial function on one level.

    Parameters
    ----------
    level : Level
        The families.
    trial_function : ConstrainedAnsatz
        Trial function ``u_theta``.
    problem : EllipticProblem
        The problem.
    names : Sequence[str]
        Terms to compute, among ``ALL_TERMS``.

    Returns
    -------
    terms : dict[str, jax.Array]
        The terms asked, in the order of ``names``, each of shape ``()``;
        ``rho^2`` and ``rho_P^2`` sum over all patches of their family.
    """
    terms = {}
    if any(name in FAMILY_TERMS["crisscross"] for name in names):
        space = level.crisscross_space
        trial = evaluate_network(trial_function, space.points)
        residual = jnp.take(
            compute_meshfree_patch_residual(space, trial),
            space.family.interior_patches,
            axis=-2,
        )
        terms[ETA_CRISSCROSS] = riesz_loss(residual, level.crisscross_factor)
        if RHO_CRISSCROSS in names:
            flux_divergence = evaluate_network_flux_divergence(
                trial_function, problem.diffusion, space.points
            )
            terms[RHO_CRISSCROSS] = jnp.sum(
                compute_meshfree_patch_residual_terms(space, trial, flux_divergence)
            )
    if any(name in FAMILY_TERMS["linf"] for name in names):
        space = level.raw_space
        interior = space.family.interior_patches
        trial = evaluate_network(trial_function, space.points)
        if MF in names or ETA_PHI in names:
            residual = jnp.take(
                compute_meshfree_patch_residual(space, trial), interior, axis=-2
            )
            terms[MF] = mf_vpinn_loss(residual, level.interior_areas)
            terms[ETA_PHI] = riesz_loss(residual, level.raw_factor)
        if ETA_PSI in names:
            residual = jnp.take(
                compute_meshfree_patch_residual(level.normalised_space, trial),
                interior,
                axis=-2,
            )
            terms[ETA_PSI] = riesz_loss(residual, level.normalised_factor)
        if RHO in names:
            flux_divergence = evaluate_network_flux_divergence(
                trial_function, problem.diffusion, space.points
            )
            terms[RHO] = jnp.sum(
                compute_meshfree_patch_residual_terms(space, trial, flux_divergence)
            )
    return {name: terms[name] for name in names}


def loss_label(loss: Loss) -> str:
    """
    Legend label of a loss.

    Parameters
    ----------
    loss : Loss
        The loss.

    Returns
    -------
    label : str
        The sum of its terms in LaTeX.
    """
    return "$" + " + ".join(name.strip("$") for name in loss.terms) + "$"


def plot_level_comparison(
    histories: Mapping[str, TrainingHistory], cells_per_side: int
) -> Figure:
    """
    Relative energy error and efficiency index of the runs of one level.

    Parameters
    ----------
    histories : Mapping[str, TrainingHistory]
        Histories of trainings with a ``create_training_monitor`` monitor, by
        legend label.
    cells_per_side : int
        Number of grid cells per side ``n`` of the level.

    Returns
    -------
    figure : Figure
        Two panels sharing the step axis: ``||grad(u - u_theta)||_0 /
        ||grad u||_0`` and ``sqrt(L) / ||grad(u - u_theta)||_0`` of every run
        at the monitored steps.
    """
    figure, (error_axis, index_axis) = plt.subplots(
        2, 1, figsize=(9.0, 9.0), sharex=True, constrained_layout=True
    )
    for index, (label, history) in enumerate(histories.items()):
        steps = np.asarray(history.monitored_steps)
        record = history.monitored
        error_axis.semilogy(
            steps,
            np.asarray(record.relative_energy_error),
            label=label,
            **line_style(index),
        )
        index_axis.semilogy(
            steps, np.asarray(record.robustness), label=label, **line_style(index)
        )
    index_axis.axhline(1.0, color="0.5", linewidth=1.0)
    error_axis.set_ylabel(rf"${_ERROR} / \|\nabla u\|_0$")
    error_axis.set_title("Relative energy error")
    index_axis.set_ylabel(rf"$\sqrt{{\mathcal{{L}}}} / {_ERROR}$")
    index_axis.set_title("Efficiency index of the loss")
    index_axis.set_xlabel("step")
    figure.suptitle(rf"$n = {cells_per_side}$ ($s = 1 / {cells_per_side}$)")
    error_axis.legend(ncol=2, fontsize="small")
    for axis in (error_axis, index_axis):
        show_grid(axis)
        axis.tick_params(labelbottom=True)
    return figure


def _set_level_axis(axis: Axes, grid_sizes: Sequence[int]) -> None:
    """
    Logarithmic ``n`` axis with a tick at every level.

    Parameters
    ----------
    axis : Axes
        Axes whose horizontal axis is ``n``.
    grid_sizes : Sequence[int]
        The levels.
    """
    axis.set_xscale("log")
    axis.set_xticks(grid_sizes, [str(n) for n in grid_sizes])
    axis.xaxis.set_minor_locator(NullLocator())
    axis.set_xlabel(r"cells per side $n$ ($s = 1 / n$)")
    show_grid(axis)


def plot_convergence(rows: Sequence[Mapping[str, Any]]) -> Figure:
    """
    Errors and efficiency indices of every loss against the number of cells.

    Parameters
    ----------
    rows : Sequence[Mapping[str, Any]]
        The rows of ``summary.csv``.

    Returns
    -------
    figure : Figure
        Two log-log panels: the relative energy error and the efficiency
        index at the end of every training, one line per loss, with a
        reference line proportional to ``s`` in the second.
    """
    figure, (error_axis, index_axis) = plt.subplots(
        1, 2, figsize=(12.0, 4.8), constrained_layout=True
    )
    grid_sizes = sorted({row["cells_per_side"] for row in rows})
    for index, run in enumerate(dict.fromkeys(row["run"] for row in rows)):
        own = [row for row in rows if row["run"] == run]
        for axis, key in (
            (error_axis, "relative_energy_error"),
            (index_axis, "efficiency_index"),
        ):
            axis.loglog(
                [row["cells_per_side"] for row in own],
                [row[key] for row in own],
                marker="o",
                label=loss_label(LOSSES[run]),
                **line_style(index),
            )
    first = [row for row in rows if row["cells_per_side"] == grid_sizes[0]]
    anchor = max(row["efficiency_index"] for row in first)
    index_axis.loglog(
        grid_sizes,
        [anchor * grid_sizes[0] / n for n in grid_sizes],
        color="0.5",
        linewidth=1.0,
        label=r"$\propto s$",
    )
    index_axis.axhline(1.0, color="0.5", linewidth=1.0, linestyle=":")
    error_axis.set_ylabel(rf"${_ERROR} / \|\nabla u\|_0$")
    error_axis.set_title("Relative energy error at the end of training")
    index_axis.set_ylabel(rf"$\sqrt{{\mathcal{{L}}}} / {_ERROR}$")
    index_axis.set_title("Efficiency index at the end of training")
    for axis in (error_axis, index_axis):
        _set_level_axis(axis, grid_sizes)
        axis.legend(fontsize="small")
    return figure


def _index_table(rows: Sequence[Mapping[str, Any]], key: str) -> list[dict[str, Any]]:
    """
    One quantity of every run on every level, with a row per run.

    Parameters
    ----------
    rows : Sequence[Mapping[str, Any]]
        The rows of ``summary.csv``.
    key : str
        Column of the quantity.

    Returns
    -------
    table : list[dict[str, Any]]
        For every run, its name and the quantity on every level, under
        ``"n = <n>"``.
    """
    return [
        {
            "run": run,
            **{
                f"n = {row['cells_per_side']}": row[key]
                for row in rows
                if row["run"] == run
            },
        }
        for run in dict.fromkeys(row["run"] for row in rows)
    ]


def _print_table(table: Sequence[Mapping[str, Any]]) -> None:
    """
    Print a table of runs against levels.

    Parameters
    ----------
    table : Sequence[Mapping[str, Any]]
        Rows from ``_index_table``.
    """
    columns = [key for key in table[0] if key != "run"]
    print(f"{'run':>18s}" + "".join(f"{column:>11s}" for column in columns))
    for row in table:
        print(
            f"{row['run']:>18s}" + "".join(f"{row[column]:11.3e}" for column in columns)
        )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """
    Write rows with the same keys as a CSV file.

    Parameters
    ----------
    path : Path
        File to write.
    rows : Sequence[Mapping[str, Any]]
        The rows; the keys of the first are the header.
    """
    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(
    grid_sizes: tuple[int, ...] = (2, 3, 4, 5, 6, 7, 8, 9, 10, 12),
    losses: tuple[str, ...] = DEFAULT_LOSSES,
    width: int = 32,
    depth: int = 3,
    num_steps: int = 6000,
    learning_rate: float = 1e-3,
    final_learning_rate: float = 1e-4,
    seed: int = 42067,
    quadrature_degree: int = 2,
    monitor_every: int = 25,
    error_cells_per_side: int = 32,
    error_quadrature_degree: int = 6,
) -> None:
    """
    Train one network per loss and level and compare errors and estimators.

    Parameters
    ----------
    grid_sizes : tuple[int, ...]
        Numbers of grid cells per side ``n`` of the levels, increasing.
    losses : tuple[str, ...]
        Keys of ``LOSSES`` to train on every level.
    width : int
        Hidden-layer width of the network.
    depth : int
        Number of hidden layers.
    num_steps : int
        Number of Adam steps per training.
    learning_rate : float
        Initial Adam learning rate.
    final_learning_rate : float
        Adam learning rate after ``num_steps`` steps, reached by exponential
        decay.
    seed : int
        Seed of the initial network weights, shared by all trainings.
    quadrature_degree : int
        Exactness degree of the rule on the local triangles of every patch.
    monitor_every : int
        Number of steps between evaluations of the energy error.
    error_cells_per_side : int
        Cells per side of the criss-cross mesh on which the errors are
        measured.
    error_quadrature_degree : int
        Exactness degree of the rule on the cells of that mesh.
    """
    element = LagrangeTriangleP1(reference_cell=ReferenceTriangle())
    error_mesh = create_crisscross_unit_square_mesh(error_cells_per_side)
    error_basis = create_cell_basis(
        create_finite_element_space(error_mesh, element),
        element.reference_cell.create_quadrature(error_quadrature_degree),
    )
    # The mesh only tags the boundary of the Dirichlet condition; the patch
    # spaces use the coefficient functions and the lift the data.
    problem = create_simple_elliptic_problem(error_mesh)
    l2_norm = compute_solution_norm(error_basis, problem, l2_error_functional)
    energy_norm = compute_solution_norm(error_basis, problem, energy_error_functional)
    distance = ExactDistanceFunction("unit_square")
    (dirichlet_condition,) = problem.dirichlet_conditions
    lift = WachspressLift(dirichlet_condition.value)

    def trial_network(model: eqx.Module) -> ConstrainedAnsatz:
        """
        Trial function of a free network.

        Parameters
        ----------
        model : eqx.Module
            Free network ``N_theta``.

        Returns
        -------
        trial_network : ConstrainedAnsatz
            ``u_theta = zeta N_theta + g_bar``.
        """
        return ConstrainedAnsatz(model, distance, lift)

    def loss_terms(model: eqx.Module, level: Level, run: str) -> dict[str, jax.Array]:
        """
        Named terms of the loss of a run on a level.

        Parameters
        ----------
        model : eqx.Module
            Free network.
        level : Level
            The families.
        run : str
            Key of ``LOSSES``.

        Returns
        -------
        terms : dict[str, jax.Array]
            The terms whose sum is the loss.
        """
        return compute_loss_terms(
            level, trial_network(model), problem, LOSSES[run].terms
        )

    def energy_error(model: eqx.Module) -> jax.Array:
        """
        Energy error of the trial function of a free network.

        Parameters
        ----------
        model : eqx.Module
            Free network.

        Returns
        -------
        error : jax.Array
            ``||grad(u - u_theta)||_0``, shape ``()``.
        """
        return compute_network_error(
            trial_network(model), error_basis, problem, energy_error_functional
        )

    output_directory = create_timestamped_output_directory(
        "experiments", "11_paper_convergence_study"
    )
    print(
        f"Problem {problem.name}; levels n = {grid_sizes}; "
        f"losses {', '.join(losses)}; {num_steps} Adam steps per training, "
        f"learning rate {learning_rate:g} -> {final_learning_rate:g}."
    )
    rows = []
    for n in grid_sizes:
        level = create_level(n, problem, quadrature_degree)
        level_directory = output_directory / f"n{n}"
        linf_family = level.raw_space.family
        crisscross_family = level.crisscross_space.family
        s_min, s_max = level.normalised_space.covering_bounds
        print(
            f"\n=== n = {n}, s = 1/{n} ===\n"
            f"l^inf family: {linf_family.interior_patches.shape[-1]} interior "
            f"patches (test functions) and "
            f"{np.count_nonzero(np.asarray(linf_family.boundary_patch_mask))} side "
            f"and corner patches; S in [{s_min:.3f}, {s_max:.3f}].\n"
            f"Criss-cross family: {crisscross_family.interior_patches.shape[-1]} "
            f"interior patches and "
            f"{np.count_nonzero(np.asarray(crisscross_family.boundary_patch_mask))} "
            f"boundary patches."
        )

        histories = {}
        level_rows = []
        for run in losses:
            run_terms = functools.partial(loss_terms, level=level, run=run)

            def loss_fn(model: eqx.Module, run_terms=run_terms) -> jax.Array:
                """
                Sum of the loss terms of the current run.

                Parameters
                ----------
                model : eqx.Module
                    Free network.
                run_terms : Callable[[eqx.Module], dict[str, jax.Array]]
                    Terms of the loss of the current run.

                Returns
                -------
                loss : jax.Array
                    Loss, shape ``()``.
                """
                return sum(run_terms(model).values())

            start = time.perf_counter()
            network, history = train_adam(
                loss_fn,
                MultiLayerPerceptron(width=width, depth=depth, key=jr.PRNGKey(seed)),
                num_steps=num_steps,
                learning_rate=learning_rate,
                final_learning_rate=final_learning_rate,
                monitor=create_training_monitor(run_terms, energy_error, energy_norm),
                monitor_every=monitor_every,
            )
            elapsed = time.perf_counter() - start
            histories[loss_label(LOSSES[run])] = history

            # Every estimator of the study on the trained network.
            trial_function = trial_network(network)
            error = float(energy_error(network))
            terms = {
                name: float(value)
                for name, value in compute_loss_terms(
                    level, trial_function, problem, ALL_TERMS
                ).items()
            }
            loss = sum(terms[name] for name in LOSSES[run].terms)
            family = (
                crisscross_family if LOSSES[run].family == "crisscross" else linf_family
            )
            row = {
                "cells_per_side": n,
                "run": run,
                "patches": family.number_of_patches,
                "test_functions": int(family.interior_patches.shape[-1]),
                "time_s": elapsed,
                "relative_l2_error": float(
                    compute_network_error(
                        trial_function, error_basis, problem, l2_error_functional
                    )
                    / l2_norm
                ),
                "energy_error": error,
                "relative_energy_error": error / float(energy_norm),
                "sqrt_loss": np.sqrt(loss),
                "efficiency_index": np.sqrt(loss) / error,
                **{COLUMNS[name][0]: np.sqrt(terms[name]) for name in ALL_TERMS},
            }
            level_rows.append(row)
            print(
                f"[n = {n}, {run:18s}] {elapsed:7.1f}s | rel. energy error "
                f"{row['relative_energy_error']:.3e} | sqrt(L) / error "
                f"{row['efficiency_index']:.3f}"
            )

            figures = {
                "training": plot_training_quantities(history),
                "loss_terms": plot_loss_terms(history),
                "robustness": plot_robustness(history),
                "solution": plot_network_solution(trial_function, error_mesh, problem),
            }
            for name, figure in figures.items():
                save_plot(figure, level_directory / f"{run}_{name}.png")

        print(f"\nn = {n}: estimators of every trained network, relative to its error:")
        print(
            f"{'run':>18s} {'rel. error':>11s} {'sqrt(L)':>9s}"
            + "".join(f" {COLUMNS[name][1]:>10s}" for name in ALL_TERMS)
        )
        for row in level_rows:
            print(
                f"{row['run']:>18s} {row['relative_energy_error']:11.3e} "
                f"{row['efficiency_index']:9.3e}"
                + "".join(
                    f" {row[COLUMNS[name][0]] / row['energy_error']:10.3e}"
                    for name in ALL_TERMS
                )
            )
        _write_csv(level_directory / "summary.csv", level_rows)
        save_plot(
            plot_level_comparison(histories, n), level_directory / "comparison.png"
        )

        # The tables and plot of all levels so far, updated after every level.
        rows.extend(level_rows)
        _write_csv(output_directory / "summary.csv", rows)
        _write_csv(
            output_directory / "efficiency_index.csv",
            _index_table(rows, "efficiency_index"),
        )
        save_plot(plot_convergence(rows), output_directory / "convergence.png")

    print(
        "\nEfficiency index sqrt(L) / ||grad(u - u_theta)||_0 at the end of training:"
    )
    _print_table(_index_table(rows, "efficiency_index"))
    print("\nRelative energy error at the end of training:")
    _print_table(_index_table(rows, "relative_energy_error"))
    print(f"Saved plots and tables to {output_directory}/")


if __name__ == "__main__":
    main()
