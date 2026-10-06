"""Uniform convergence study of the MF-VPINN with the estimator loss.

On l^inf cone families of the unit square with spacing ``1/n``, a fresh network
is trained for each ``n`` with the loss ``eta_P^2 + lambda sum_i eta_R,i^2``,
for the Poisson problem with solution sin(pi x) sin(pi y) and the Dirichlet
condition imposed by the ansatz. Each level records the errors, the estimator
terms, and the ratio of the estimator to the energy error; the history is
plotted against the number of test functions, and every level's training
quantities, loss terms, robustness, solution, and cellwise errors are plotted,
all under ``experiments/``.

The trial set does not change with ``n``, so the error is limited by the
training as well as by the test space; the study checks that the estimator
stays comparable to the energy error on every level.

Run with:

    uv run python examples/05_convergence_study.py
"""

import time
from collections.abc import Mapping
from pathlib import Path

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
from jax_fem.problem import create_poisson_sin_sin_problem

from vpinns_jax.analysis import (
    compute_network_error,
    compute_solution_norm,
    create_training_monitor,
    plot_cell_errors,
    plot_loss_terms,
    plot_network_solution,
    plot_robustness,
    plot_training_quantities,
)
from vpinns_jax.losses import (
    compute_patch_residual,
    compute_patch_residual_terms,
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
from vpinns_jax.solvers import train_adam
from vpinns_jax.test_spaces import create_linf_grid_family, create_patch_test_space

_COLUMNS = ("test functions", "L2 error", "energy error", "eta_P", "eta_R", "estimator")


def _run_level(
    cells_per_side: int,
    network: MultiLayerPerceptron,
    output_directory: Path,
    *,
    num_steps: int,
    learning_rate: float,
    residual_weight: float,
) -> tuple[float, ...]:
    """
    Train a network on one family, measure it, and save its plots.

    Parameters
    ----------
    cells_per_side : int
        Number of grid squares per side.
    network : MultiLayerPerceptron
        Initial free network.
    output_directory : Path
        Directory of the plots, prefixed with the level.
    num_steps : int
        Number of Adam steps.
    learning_rate : float
        Adam learning rate.
    residual_weight : float
        Weight ``lambda`` of the residual terms in the loss.

    Returns
    -------
    record : tuple[float, ...]
        Number of test functions, L2 error, energy error, ``eta_P``,
        ``eta_R``, and the estimator ``(eta_P^2 + lambda eta_R^2)^(1/2)``.
    """
    family = create_linf_grid_family(
        np.ones((cells_per_side, cells_per_side), dtype=bool),
        spacing=1.0 / cells_per_side,
    )
    problem = create_poisson_sin_sin_problem(family.mesh)
    space = create_patch_test_space(family, problem)
    interior = family.interior_patches
    factor = jnp.linalg.cholesky(restrict_to_free(space.gram, interior))
    points = space.basis.physical_points
    distance = ExactDistanceFunction("unit_square")

    def squared_terms(model: eqx.Module) -> tuple[jax.Array, jax.Array]:
        """
        Squared coupled and residual terms of the constrained network.

        Parameters
        ----------
        model : eqx.Module
            Free network.

        Returns
        -------
        coupled : jax.Array
            ``eta_P^2``, shape ``()``.
        residual : jax.Array
            ``sum_i eta_R,i^2``, shape ``()``.
        """
        trial_network = ConstrainedAnsatz(model, distance)
        trial = evaluate_network(trial_network, points)
        residual = jnp.take(compute_patch_residual(space, trial), interior, axis=-2)
        flux_divergence = evaluate_network_flux_divergence(
            trial_network, problem.diffusion, points
        )
        terms = compute_patch_residual_terms(space, trial, flux_divergence)
        return riesz_loss(residual, factor), jnp.sum(terms)

    def loss_terms(model: eqx.Module) -> Mapping[str, jax.Array]:
        """
        Named terms of the estimator loss of the constrained network.

        Parameters
        ----------
        model : eqx.Module
            Free network.

        Returns
        -------
        terms : Mapping[str, jax.Array]
            ``eta_P^2`` and ``lambda sum_i eta_R,i^2``.
        """
        coupled, residual = squared_terms(model)
        return {
            r"$\eta_P^2$": coupled,
            r"$\lambda \sum_i \eta_{R,i}^2$": residual_weight * residual,
        }

    def loss_fn(model: eqx.Module) -> jax.Array:
        """
        Estimator loss of the constrained network.

        Parameters
        ----------
        model : eqx.Module
            Free network.

        Returns
        -------
        loss : jax.Array
            ``eta_P^2 + lambda sum_i eta_R,i^2``, shape ``()``.
        """
        return sum(loss_terms(model).values())

    def energy_error(model: eqx.Module) -> jax.Array:
        """
        Energy error of the constrained network.

        Parameters
        ----------
        model : eqx.Module
            Free network.

        Returns
        -------
        error : jax.Array
            ``|u - w|_A``, shape ``()``.
        """
        return compute_network_error(
            ConstrainedAnsatz(model, distance),
            space.basis,
            problem,
            energy_error_functional,
        )

    monitor = create_training_monitor(
        loss_terms,
        energy_error,
        compute_solution_norm(space.basis, problem, energy_error_functional),
    )
    network, history = train_adam(
        loss_fn,
        network,
        num_steps=num_steps,
        learning_rate=learning_rate,
        monitor=monitor,
        monitor_every=max(1, num_steps // 100),
    )
    trial_network = ConstrainedAnsatz(network, distance)
    cell_errors = {
        "$L^2$ error": compute_network_error(
            trial_network, space.basis, problem, l2_error_functional, reduce=False
        ),
        "energy error": compute_network_error(
            trial_network, space.basis, problem, energy_error_functional, reduce=False
        ),
    }
    l2 = jnp.sqrt(jnp.sum(cell_errors["$L^2$ error"] ** 2))
    energy = jnp.sqrt(jnp.sum(cell_errors["energy error"] ** 2))
    figures = {
        "training": plot_training_quantities(history),
        "loss_terms": plot_loss_terms(history),
        "robustness": plot_robustness(history),
        "solution": plot_network_solution(trial_network, family.mesh, problem),
        "cell_errors": plot_cell_errors(family.mesh, cell_errors),
    }
    for name, figure in figures.items():
        save_plot(figure, output_directory / f"n{cells_per_side}_{name}.png")
    coupled, residual = (float(jnp.sqrt(term)) for term in squared_terms(network))
    estimator = float(np.sqrt(coupled**2 + residual_weight * residual**2))
    return (interior.shape[-1], float(l2), float(energy), coupled, residual, estimator)


def main(
    grid_sizes: tuple[int, ...] = (4, 8, 16),
    width: int = 32,
    depth: int = 3,
    num_steps: int = 3000,
    learning_rate: float = 1e-3,
    residual_weight: float = 1.0,
    seed: int = 0,
) -> None:
    """
    Train on every family and record errors and estimator terms.

    Parameters
    ----------
    grid_sizes : tuple[int, ...]
        Numbers of grid squares per side.
    width : int
        Hidden-layer width of the network.
    depth : int
        Number of hidden layers.
    num_steps : int
        Number of Adam steps per level.
    learning_rate : float
        Adam learning rate.
    residual_weight : float
        Weight ``lambda`` of the residual terms in the loss.
    seed : int
        Seed of the initial network weights, shared by all levels.
    """
    print(
        "    n"
        + "".join(f"{column:>15s}" for column in _COLUMNS)
        + "   estimator/energy"
    )
    output_directory = create_timestamped_output_directory(
        "experiments", "05_convergence_study"
    )
    records = []
    for cells_per_side in grid_sizes:
        start = time.perf_counter()
        record = _run_level(
            cells_per_side,
            MultiLayerPerceptron(width=width, depth=depth, key=jr.PRNGKey(seed)),
            output_directory,
            num_steps=num_steps,
            learning_rate=learning_rate,
            residual_weight=residual_weight,
        )
        records.append(record)
        print(
            f"{cells_per_side:5d}{record[0]:15d}"
            + "".join(f"{value:15.3e}" for value in record[1:])
            + f"   {record[5] / record[2]:.3f}   ({time.perf_counter() - start:.1f}s)"
        )

    table = np.asarray(records)
    figure, axis = plt.subplots(figsize=(6, 4))
    for column in range(2, len(_COLUMNS)):
        axis.loglog(table[:, 0], table[:, column], "o-", label=_COLUMNS[column])
    axis.set_xlabel("number of test functions")
    axis.set_title("MF-VPINN with the estimator loss")
    axis.legend()
    axis.grid(True, which="both", linestyle=":", linewidth=0.5)

    save_plot(figure, output_directory / "convergence.png")
    print(f"Saved plots to {output_directory}/")


if __name__ == "__main__":
    main()
