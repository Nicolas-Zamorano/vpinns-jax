"""Loss functionals of the MF-VPINN on a smooth Poisson problem (Experiment 3).

The problem is ``-Lap u = f`` in the unit square with ``u = 0`` on the
boundary, for one of two solutions:

* ``"sin_sin"``: ``u = sin(pi x) sin(pi y)``;
* ``"boundary_layer"``: ``u = x y (1 - x) (1 - y) (exp(k x) - 1)``, with a
  layer of width about ``1 / k`` at ``x = 1``.

The trial function is ``u_theta = zeta N_theta`` with
``zeta = x (1 - x) y (1 - y)``, so ``u_theta = 0`` on the boundary.

Eight networks with the same initial weights are trained with Adam, one per
loss, with the learning rate decaying exponentially from ``1e-3`` to
``1e-4``. The first six use the ``l^inf`` family of the paper: a square
patch at every interior grid point and, at the boundary grid points, the
side and corner patches, which are not test functions but enter the
covering ``S`` and the estimator ``rho``. The last two use the criss-cross
family, whose weights sum to one.

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

Here ``L_MF = (1 / N) sum_i r_i^2 / |P_i|`` is the loss of the original
MF-VPINN, with the residuals ``r_i = l(phi_i) - a(u_theta, phi_i)`` of the raw
weights of the ``N`` interior patches. ``eta_phi`` and ``eta_psi`` are the dual
norms ``(r^T G^{-1} r)^(1/2)`` of the residual over the span of the raw
weights ``phi_i`` and of the normalised weights ``psi_i = phi_i / S`` of the
interior patches, and ``eta_P`` is the same over the criss-cross weights.
The estimator ``rho^2 = sum_i c_2(P_i)^2 h_i^2 ||phi_i^(1/2) (f + Lap u_theta)||^2``
sums over all patches of the family, boundary patches included; for the
``l^inf`` family the constants ``c_2`` are computed by eigenvalue problems on
the reference configurations.

The losses are integrated on the triangles of the criss-cross triangulation
``T_s``, on which every weight and the covering ``S`` are affine, with the
degree-4 rule; every reported value, including those recorded during the
training, is recomputed with the degree-8 rule, which integrates the rational
``psi_i`` more accurately. The errors are measured on the criss-cross mesh
with ``error_cells_per_side`` cells per side and the degree-6 rule.

At every monitored step, every estimator is recorded for every run, and the
two-sided bounds of the paper are checked:

* ``eta_psi <= ||grad(u - u_theta)||_0 <= ||Pi_psi|| eta_psi + sqrt(2) rho``,
  with the computed norm ``||Pi_psi||_h`` and with the a priori bound
  ``N_ov K_psi``;
* ``eta_P <= ||grad(u - u_theta)||_0 <= ||Pi_P|| eta_P + rho_P``, with the
  computed norm ``||Pi_P||_h``.

The lower bounds hold exactly, so a computed ``eta / error`` above one
reveals a quadrature error. The norms ``||Pi||_h`` are suprema over the
continuous piecewise linear functions on a refinement of ``T_s``.

For every run the script plots the training quantities, the loss terms, the
robustness ``sqrt(L) / ||grad(u - u_theta)||_0``, and the exact and network
solutions. It prints and saves the table of the final errors, the final,
smallest and largest efficiency index, the final estimators, and the extreme
bound ratios along the training as ``summary.csv``, and plots the relative
error, the robustness and the bound ratios of all runs together, all under
``experiments/``.

Run with:

    uv run python examples/10_paper_singular_poisson.py
"""

import csv
import functools
import math
import time
from collections.abc import Mapping
from typing import NamedTuple

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
from jax_fem.function import FiniteElementFunctionEvaluation
from jax_fem.mesh import TriangleMesh
from jax_fem.problem import (
    EllipticProblem,
    create_poisson_exponential_problem,
    create_poisson_sin_sin_problem,
)
from jax_fem.reference_cell import ReferenceTriangle
from jax_fem.space import create_cell_basis, create_finite_element_space
from matplotlib.figure import Figure

from vpinns_jax.analysis import (
    ErrorBound,
    compute_bound_ratios,
    compute_network_error,
    compute_quasi_interpolant_norm,
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
    evaluate_network,
    evaluate_network_flux_divergence,
)
from vpinns_jax.solvers import TrainingHistory, train_adam
from vpinns_jax.test_spaces import (
    LINF_CONFIGURATIONS,
    MeshfreePatchFamily,
    MeshfreePatchTestSpace,
    compute_linf_poincare_constants,
    create_crisscross_family,
    create_crisscross_unit_square_mesh,
    create_linf_vertex_family,
    create_meshfree_patch_test_space,
)
from vpinns_jax.test_spaces.poincare_constants import round_up

# Loss terms and estimators, named as in the paper; the names label the plots.
MF = r"$\mathcal{L}_{\mathrm{MF}}$"
ETA_PHI = r"$\eta_\varphi^2$"
ETA_PSI = r"$\eta_\psi^2$"
RHO = r"$\rho^2$"
ETA_CRISSCROSS = r"$\eta_{\mathcal{P}}^2$"
RHO_CRISSCROSS = r"$\rho_{\mathcal{P}}^2$"

# Square-root estimators reported in the summary, by column name.
ESTIMATOR_COLUMNS = {
    # "sqrt_L_MF": MF,
    # "eta_phi": ETA_PHI,
    "eta_psi": ETA_PSI,
    "rho": RHO,
    # "eta_crisscross": ETA_CRISSCROSS,
    # "rho_crisscross": RHO_CRISSCROSS,
}

# Overlap N_ov and ||grad psi_c||_inf s of the l^inf family (Lemma 5.1).
LINF_OVERLAP = 4
LINF_GRADIENT_BOUND = math.sqrt(5.0)


class Loss(NamedTuple):
    """
    A loss of the comparison.

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
    # "mf": Loss("linf", (MF,)),
    # "mf_rho": Loss("linf", (MF, RHO)),
    # "eta_phi": Loss("linf", (ETA_PHI,)),
    # "eta_phi_rho": Loss("linf", (ETA_PHI, RHO)),
    # "eta_psi": Loss("linf", (ETA_PSI,)),
    "eta_psi_rho": Loss("linf", (ETA_PSI, RHO)),
    # "eta_crisscross": Loss("crisscross", (ETA_CRISSCROSS,)),
    # "eta_crisscross_rho": Loss("crisscross", (ETA_CRISSCROSS, RHO_CRISSCROSS)),
}


class Spaces(NamedTuple):
    """
    Test spaces of both families integrated with one rule.

    Attributes
    ----------
    raw : MeshfreePatchTestSpace
        ``l^inf`` family with the raw weights ``phi_i``.
    normalised : MeshfreePatchTestSpace
        ``l^inf`` family with the normalised weights ``psi_i``; it shares the
        points of ``raw``.
    crisscross : MeshfreePatchTestSpace
        Criss-cross family.
    raw_factor : jax.Array
        Cholesky factor of the interior Gram matrix of ``raw``.
    normalised_factor : jax.Array
        Cholesky factor of the interior Gram matrix of ``normalised``.
    crisscross_factor : jax.Array
        Cholesky factor of the interior Gram matrix of ``crisscross``.
    """

    raw: MeshfreePatchTestSpace
    normalised: MeshfreePatchTestSpace
    crisscross: MeshfreePatchTestSpace
    raw_factor: jax.Array
    normalised_factor: jax.Array
    crisscross_factor: jax.Array


def create_problem(name: str, mesh: TriangleMesh, k: float) -> EllipticProblem:
    """
    The Poisson problem of the experiment.

    Parameters
    ----------
    name : str
        ``"sin_sin"`` or ``"boundary_layer"``.
    mesh : TriangleMesh
        Mesh of the unit square carrying the boundary tags.
    k : float
        Rate ``k`` of the boundary layer.

    Returns
    -------
    problem : EllipticProblem
        ``-Lap u = f`` with ``u = 0`` on the boundary.
    """
    if name == "sin_sin":
        return create_poisson_sin_sin_problem(mesh)
    if name == "boundary_layer":
        return create_poisson_exponential_problem(mesh, exponential_constant=k)
    raise ValueError(f"problem must be 'sin_sin' or 'boundary_layer', got {name!r}.")


def interior_cholesky_factor(space: MeshfreePatchTestSpace) -> jax.Array:
    """
    Cholesky factor of the Gram matrix of the test functions.

    Parameters
    ----------
    space : MeshfreePatchTestSpace
        Patch test space.

    Returns
    -------
    factor : jax.Array
        Lower factor of the energy Gram matrix restricted to the interior
        patches, shape ``(N_interior, N_interior)``.
    """
    return jnp.linalg.cholesky(
        restrict_to_free(space.gram, space.family.interior_patches)
    )


def create_spaces(
    linf_family: MeshfreePatchFamily,
    crisscross_family: MeshfreePatchFamily,
    problem: EllipticProblem,
    quadrature_degree: int,
) -> Spaces:
    """
    Test spaces of both families with one rule on the local triangles.

    Parameters
    ----------
    linf_family : MeshfreePatchFamily
        ``l^inf`` family.
    crisscross_family : MeshfreePatchFamily
        Criss-cross family.
    problem : EllipticProblem
        Problem providing the coefficients.
    quadrature_degree : int
        Exactness degree of the rule on every local triangle.

    Returns
    -------
    spaces : Spaces
        The three test spaces and their Cholesky factors.
    """
    raw = create_meshfree_patch_test_space(
        linf_family, problem, quadrature_degree=quadrature_degree
    )
    normalised = create_meshfree_patch_test_space(
        linf_family, problem, quadrature_degree=quadrature_degree, normalise=True
    )
    crisscross = create_meshfree_patch_test_space(
        crisscross_family, problem, quadrature_degree=quadrature_degree
    )
    return Spaces(
        raw=raw,
        normalised=normalised,
        crisscross=crisscross,
        raw_factor=interior_cholesky_factor(raw),
        normalised_factor=interior_cholesky_factor(normalised),
        crisscross_factor=interior_cholesky_factor(crisscross),
    )


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
    return "$" + " + ".join(term.strip("$") for term in loss.terms) + "$"


def plot_comparison(histories: Mapping[str, TrainingHistory]) -> Figure:
    """
    Relative energy error and robustness of several runs.

    Parameters
    ----------
    histories : Mapping[str, TrainingHistory]
        Histories of trainings with a ``create_training_monitor`` monitor, by
        legend label.

    Returns
    -------
    figure : Figure
        Two panels sharing the step axis: ``||grad(u - u_theta)||_0 /
        ||grad u||_0`` and ``sqrt(L) / ||grad(u - u_theta)||_0`` of every run
        at the monitored steps.
    """
    figure, (error_axis, robustness_axis) = plt.subplots(
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
        robustness_axis.semilogy(
            steps, np.asarray(record.robustness), label=label, **line_style(index)
        )
    robustness_axis.axhline(1.0, color="0.5", linewidth=1.0)
    error_axis.set_ylabel(r"$\|\nabla(u - u_\theta)\|_0 / \|\nabla u\|_0$")
    error_axis.set_title("Relative energy error")
    robustness_axis.set_ylabel(r"$\sqrt{\mathcal{L}} / \|\nabla(u - u_\theta)\|_0$")
    robustness_axis.set_title("Efficiency index of the loss")
    robustness_axis.set_xlabel("step")
    error_axis.legend(ncol=2, fontsize="small")
    for axis in (error_axis, robustness_axis):
        show_grid(axis)
        axis.tick_params(labelbottom=True)
    return figure


def plot_bound_checks(
    histories: Mapping[str, TrainingHistory], bounds: Mapping[str, ErrorBound]
) -> Figure:
    """
    Both sides of the error bounds relative to the error along every run.

    Parameters
    ----------
    histories : Mapping[str, TrainingHistory]
        Histories whose records contain every estimator, by legend label.
    bounds : Mapping[str, ErrorBound]
        Bounds to check, by panel title; one row of panels each.

    Returns
    -------
    figure : Figure
        For every bound, ``eta / error``, which the lower bound keeps at most
        one, and ``(a eta + b rho) / error``, which the upper bound keeps at
        least one, against the step.
    """
    figure, axes = plt.subplots(
        len(bounds),
        2,
        figsize=(12.0, 3.6 * len(bounds)),
        sharex=True,
        squeeze=False,
        constrained_layout=True,
    )
    for row, (title, bound) in enumerate(bounds.items()):
        lower_axis, upper_axis = axes[row]
        for index, (label, history) in enumerate(histories.items()):
            ratios = compute_bound_ratios(bound, history.monitored)
            steps = np.asarray(history.monitored_steps)
            lower_axis.plot(
                steps, np.asarray(ratios.lower), label=label, **line_style(index)
            )
            upper_axis.semilogy(
                steps, np.asarray(ratios.upper), label=label, **line_style(index)
            )
        residual_term = bound.residual_term.strip("$").removesuffix("^2")
        estimator = bound.estimator.strip("$").removesuffix("^2")
        lower_axis.set_title(f"Lower bound, {title}")
        lower_axis.set_ylabel(f"${residual_term}" + r" / \|\nabla(u - u_\theta)\|_0$")
        upper_axis.set_title(f"Upper bound, {title}")
        upper_axis.set_ylabel(
            f"$({bound.quasi_interpolant_norm:.3g}\\,{residual_term} + "
            f"{bound.remainder_constant:.3g}\\,{estimator})"
            r" / \|\nabla(u - u_\theta)\|_0$"
        )
        for axis in (lower_axis, upper_axis):
            axis.axhline(1.0, color="0.3", linewidth=1.0)
            show_grid(axis)
    for axis in axes[-1]:
        axis.set_xlabel("step")
    axes[0, 0].legend(ncol=2, fontsize="small")
    return figure


def main(
    problem_name: str = "sin_sin",
    exponential_constant: float = 16.0,
    cells_per_side: int = 16,
    width: int = 32,
    depth: int = 3,
    num_steps: int = 6000,
    learning_rate: float = 1e-3,
    final_learning_rate: float = 1e-4,
    seed: int = 0,
    quadrature_degree: int = 6,
    report_quadrature_degree: int = 8,
    monitor_every: int = 25,
    error_cells_per_side: int = 32,
    error_quadrature_degree: int = 6,
    interpolant_refinement: int = 8,
    log_every: int = 0,
    runs: tuple[str, ...] = tuple(LOSSES),
) -> None:
    """
    Train one network per loss and compare errors, estimators, and bounds.

    Parameters
    ----------
    problem_name : str
        ``"sin_sin"`` or ``"boundary_layer"``.
    exponential_constant : float
        Rate ``k`` of the boundary layer.
    cells_per_side : int
        Number of grid cells per side ``n`` of both families; the spacing is
        ``s = 1 / n``.
    width : int
        Hidden-layer width of the network.
    depth : int
        Number of hidden layers.
    num_steps : int
        Number of Adam steps.
    learning_rate : float
        Initial Adam learning rate.
    final_learning_rate : float
        Adam learning rate after ``num_steps`` steps, reached by exponential
        decay.
    seed : int
        Seed of the initial network weights, shared by all runs.
    quadrature_degree : int
        Exactness degree of the training rule on every local triangle.
    report_quadrature_degree : int
        Exactness degree of the rule of every reported and monitored value.
    monitor_every : int
        Number of steps between evaluations of the errors and estimators.
    error_cells_per_side : int
        Cells per side of the criss-cross mesh on which the errors are
        measured.
    error_quadrature_degree : int
        Exactness degree of the rule on the cells of that mesh.
    interpolant_refinement : int
        Cells per grid cell of the fine mesh of ``||Pi||_h``.
    log_every : int
        Show a progress bar updated every this many steps; ``0`` hides it.
    runs : tuple[str, ...]
        Keys of ``LOSSES`` to train.
    """
    element = LagrangeTriangleP1(reference_cell=ReferenceTriangle())
    error_mesh = create_crisscross_unit_square_mesh(error_cells_per_side)
    error_basis = create_cell_basis(
        create_finite_element_space(error_mesh, element),
        element.reference_cell.create_quadrature(error_quadrature_degree),
    )
    problem = create_problem(problem_name, error_mesh, exponential_constant)
    l2_norm = compute_solution_norm(error_basis, problem, l2_error_functional)
    energy_norm = compute_solution_norm(error_basis, problem, energy_error_functional)
    distance = ExactDistanceFunction("unit_square")

    linf_family = create_linf_vertex_family(cells_per_side)
    crisscross_family = create_crisscross_family(cells_per_side)
    training = create_spaces(linf_family, crisscross_family, problem, quadrature_degree)
    report = create_spaces(
        linf_family, crisscross_family, problem, report_quadrature_degree
    )
    linf_interior = linf_family.interior_patches
    linf_areas = jnp.take(linf_family.patch_areas, linf_interior)

    # Constants of the bounds.
    constants = {
        name: compute_linf_poincare_constants(name) for name in LINF_CONFIGURATIONS
    }
    poincare_bound = max(round_up(c.unweighted, 3) for c in constants.values())
    a_priori_norm = LINF_OVERLAP * LINF_GRADIENT_BOUND * poincare_bound
    s_min, s_max = report.normalised.covering_bounds
    remainder_constant = math.sqrt(s_max) / s_min
    psi_norm = compute_quasi_interpolant_norm(
        linf_family,
        problem,
        cells_per_side,
        normalise=True,
        refinement=interpolant_refinement,
        quadrature_degree=report_quadrature_degree,
    )
    crisscross_norm = compute_quasi_interpolant_norm(
        crisscross_family,
        problem,
        cells_per_side,
        normalise=False,
        refinement=interpolant_refinement,
        quadrature_degree=report_quadrature_degree,
    )
    bounds = {
        r"$\|\Pi_\psi\|_h$": ErrorBound(ETA_PSI, RHO, psi_norm, remainder_constant),
        r"$N_{ov} K_\psi$": ErrorBound(ETA_PSI, RHO, a_priori_norm, remainder_constant),
        r"$\|\Pi_{\mathcal{P}}\|_h$": ErrorBound(
            ETA_CRISSCROSS, RHO_CRISSCROSS, crisscross_norm, 1.0
        ),
    }

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
            ``u_theta = zeta N_theta``.
        """
        return ConstrainedAnsatz(model, distance)

    def rho_squared(
        space: MeshfreePatchTestSpace,
        model: eqx.Module,
        trial: FiniteElementFunctionEvaluation,
    ) -> jax.Array:
        """
        Squared residual estimator over all patches of a family.

        Parameters
        ----------
        space : MeshfreePatchTestSpace
            Patch test space of the family.
        model : eqx.Module
            Free network.
        trial : FiniteElementFunctionEvaluation
            Values and gradients of the trial function at ``space.points``.

        Returns
        -------
        rho_squared : jax.Array
            ``rho(u_theta)^2``, shape ``()``.
        """
        flux_divergence = evaluate_network_flux_divergence(
            trial_network(model), problem.diffusion, space.points
        )
        return jnp.sum(
            compute_meshfree_patch_residual_terms(space, trial, flux_divergence)
        )

    def linf_terms(
        spaces: Spaces, model: eqx.Module, with_rho: bool
    ) -> dict[str, jax.Array]:
        """
        Loss terms of the ``l^inf`` family.

        Parameters
        ----------
        spaces : Spaces
            Test spaces of the rule to use.
        model : eqx.Module
            Free network.
        with_rho : bool
            Also compute ``rho^2``, which needs the Laplacian of the trial.

        Returns
        -------
        terms : dict[str, jax.Array]
            ``L_MF``, ``eta_phi^2``, ``eta_psi^2``, and ``rho^2`` if asked.
        """
        trial = evaluate_network(trial_network(model), spaces.raw.points)
        raw_residual = jnp.take(
            compute_meshfree_patch_residual(spaces.raw, trial), linf_interior, axis=-2
        )
        normalised_residual = jnp.take(
            compute_meshfree_patch_residual(spaces.normalised, trial),
            linf_interior,
            axis=-2,
        )
        terms = {
            MF: mf_vpinn_loss(raw_residual, linf_areas),
            ETA_PHI: riesz_loss(raw_residual, spaces.raw_factor),
            ETA_PSI: riesz_loss(normalised_residual, spaces.normalised_factor),
        }
        if with_rho:
            terms[RHO] = rho_squared(spaces.raw, model, trial)
        return terms

    def crisscross_terms(
        spaces: Spaces, model: eqx.Module, with_rho: bool
    ) -> dict[str, jax.Array]:
        """
        Loss terms of the criss-cross family.

        Parameters
        ----------
        spaces : Spaces
            Test spaces of the rule to use.
        model : eqx.Module
            Free network.
        with_rho : bool
            Also compute ``rho_P^2``, which needs the Laplacian of the trial.

        Returns
        -------
        terms : dict[str, jax.Array]
            ``eta_P^2``, and ``rho_P^2`` if asked.
        """
        space = spaces.crisscross
        trial = evaluate_network(trial_network(model), space.points)
        residual = jnp.take(
            compute_meshfree_patch_residual(space, trial),
            space.family.interior_patches,
            axis=-2,
        )
        terms = {ETA_CRISSCROSS: riesz_loss(residual, spaces.crisscross_factor)}
        if with_rho:
            terms[RHO_CRISSCROSS] = rho_squared(space, model, trial)
        return terms

    def loss_terms(model: eqx.Module, spaces: Spaces, run: str) -> dict[str, jax.Array]:
        """
        Named terms of the loss of a run.

        Parameters
        ----------
        model : eqx.Module
            Free network.
        spaces : Spaces
            Test spaces of the rule to use.
        run : str
            Key of ``LOSSES``.

        Returns
        -------
        terms : dict[str, jax.Array]
            The terms whose sum is the loss.
        """
        loss = LOSSES[run]
        family_terms = linf_terms if loss.family == "linf" else crisscross_terms
        with_rho = RHO in loss.terms or RHO_CRISSCROSS in loss.terms
        terms = family_terms(spaces, model, with_rho)
        return {name: terms[name] for name in loss.terms}

    def all_estimators(model: eqx.Module) -> dict[str, jax.Array]:
        """
        Every squared estimator of both families with the reporting rule.

        Parameters
        ----------
        model : eqx.Module
            Free network.

        Returns
        -------
        estimators : dict[str, jax.Array]
            ``L_MF``, ``eta_phi^2``, ``eta_psi^2``, ``rho^2``, ``eta_P^2`` and
            ``rho_P^2``.
        """
        return {
            **linf_terms(report, model, True),
            **crisscross_terms(report, model, True),
        }

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
        "experiments", f"10_paper_{problem_name}"
    )
    c2_values = ", ".join(
        f"{name} {round_up(c.weighted, 3):.3f} s" for name, c in constants.items()
    )
    print(
        f"Problem {problem_name}"
        + (f" (k = {exponential_constant})" if problem_name == "boundary_layer" else "")
        + f"; n = {cells_per_side}.\n"
        f"l^inf family: {linf_interior.shape[-1]} interior patches (test functions) "
        f"and {np.count_nonzero(np.asarray(linf_family.boundary_patch_mask))} side "
        f"and corner patches; S in [{s_min:.3f}, {s_max:.3f}], "
        f"sqrt(S_max) / S_min = {remainder_constant:.3f}.\n"
        f"c_2 h: {c2_values}; C_P h <= {poincare_bound:.3f} s, so "
        f"N_ov K_psi = {a_priori_norm:.3f}.\n"
        f"||Pi_psi||_h = {psi_norm:.4f}, ||Pi_P||_h = {crisscross_norm:.4f} "
        f"(fine mesh s / {interpolant_refinement}).\n"
        f"Criss-cross family: {crisscross_family.interior_patches.shape[-1]} "
        f"interior patches, S in {report.crisscross.covering_bounds}.\n"
        f"Training rule degree {quadrature_degree}, reported values degree "
        f"{report_quadrature_degree}; Adam lr {learning_rate:g} -> "
        f"{final_learning_rate:g} over {num_steps} steps."
    )

    histories = {}
    rows = []
    for run in runs:
        run_terms = functools.partial(loss_terms, spaces=training, run=run)

        def loss_fn(model: eqx.Module, run_terms=run_terms) -> jax.Array:
            """
            Sum of the loss terms of the current run with the training rule.

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

        monitor = create_training_monitor(
            functools.partial(loss_terms, spaces=report, run=run),
            energy_error,
            energy_norm,
            estimators=all_estimators,
        )
        network = MultiLayerPerceptron(width=width, depth=depth, key=jr.PRNGKey(seed))
        start = time.perf_counter()
        network, history = train_adam(
            loss_fn,
            network,
            num_steps=num_steps,
            learning_rate=learning_rate,
            final_learning_rate=final_learning_rate,
            log_every=log_every,
            monitor=monitor,
            monitor_every=monitor_every,
        )
        elapsed = time.perf_counter() - start
        histories[loss_label(LOSSES[run])] = history

        # The last record is the trained network, with every estimator.
        records = history.monitored
        final = jax.tree.map(lambda values: values[-1], records)
        error = float(final.energy_error)
        index = np.asarray(records.robustness)
        trial_function = trial_network(network)
        row = {
            "run": run,
            "time_s": elapsed,
            "relative_l2_error": float(
                compute_network_error(
                    trial_function, error_basis, problem, l2_error_functional
                )
                / l2_norm
            ),
            "energy_error": error,
            "relative_energy_error": float(final.relative_energy_error),
            "final_index": float(index[-1]),
            "min_index": float(np.min(index)),
            "max_index": float(np.max(index)),
        }
        for column, name in ESTIMATOR_COLUMNS.items():
            row[column] = float(jnp.sqrt(final.estimators[name]))
        for column, bound in zip(
            ("psi_computed", "psi_a_priori", "crisscross_computed"),
            bounds.values(),
            strict=True,
        ):
            ratios = compute_bound_ratios(bound, records)
            row[f"max_lower_{column}"] = float(jnp.max(ratios.lower))
            row[f"min_upper_{column}"] = float(jnp.min(ratios.upper))
        rows.append(row)
        print(
            f"[{run:18s}] {elapsed:7.1f}s | rel. energy error "
            f"{row['relative_energy_error']:.3e} | index final "
            f"{row['final_index']:.3f}, min {row['min_index']:.3f}, max "
            f"{row['max_index']:.3f} | max eta_psi/err "
            f"{row['max_lower_psi_computed']:.3f}, max eta_P/err "
            f"{row['max_lower_crisscross_computed']:.3f}"
        )

        figures = {
            "training": plot_training_quantities(history),
            "loss_terms": plot_loss_terms(history),
            "robustness": plot_robustness(history),
            "solution": plot_network_solution(trial_function, error_mesh, problem),
        }
        for name, figure in figures.items():
            save_plot(figure, output_directory / f"{run}_{name}.png")

    print("\nEstimators of every trained network, relative to its energy error:")
    print(f"{'run':18s}" + "".join(f" {column:>14s}" for column in ESTIMATOR_COLUMNS))
    for row in rows:
        print(
            f"{row['run']:18s}"
            + "".join(
                f" {row[column] / row['energy_error']:14.3e}"
                for column in ESTIMATOR_COLUMNS
            )
        )
    print(
        "\nBounds along the training (lower ratios must stay <= 1, upper >= 1):\n"
        f"{'run':18s} {'eta_psi/err':>12s} {'Pi_psi,h up':>12s} "
        f"{'N_ov K up':>12s} {'eta_P/err':>12s} {'Pi_P,h up':>12s}"
    )
    for row in rows:
        print(
            f"{row['run']:18s} {row['max_lower_psi_computed']:12.3f} "
            f"{row['min_upper_psi_computed']:12.3f} "
            f"{row['min_upper_psi_a_priori']:12.3f} "
            f"{row['max_lower_crisscross_computed']:12.3f} "
            f"{row['min_upper_crisscross_computed']:12.3f}"
        )
    with open(output_directory / "summary.csv", "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    save_plot(plot_comparison(histories), output_directory / "comparison.png")
    save_plot(
        plot_bound_checks(histories, bounds), output_directory / "bound_checks.png"
    )
    print(f"Saved plots and summary.csv to {output_directory}/")


if __name__ == "__main__":
    main()
