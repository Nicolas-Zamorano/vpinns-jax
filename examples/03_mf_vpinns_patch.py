"""Mesh-free VPINN with a Shepard-normalised patch test space.

The patches are the l^inf cones centred at every grid point of the unit square,
boundary points included; the test functions are the normalised weights
``psi_i = phi_i / S`` of the interior patches. The trial function is a network
with the homogeneous Dirichlet condition imposed by the ansatz
``w = B u_theta``, for the Poisson problem with solution sin(pi x) sin(pi y).
Three losses are compared:

- decoupled: ``eta_D^2 = r^T D^{-1} r``, with ``D`` the diagonal of the energy
  Gram matrix ``P`` of the test functions;
- coupled: ``eta_P^2 = r^T P^{-1} r``, the dual norm of the residual over the
  patch space;
- estimator: ``eta_P^2 + lambda sum_i eta_R,i^2``, which adds the weighted
  strong residuals ``eta_R,i = h_i ||phi_i^(1/2) (f + lap w)||``.

For every trained network the script prints the errors, the three estimator
terms, and the ratios of ``eta_P`` and of the estimator to the energy error,
and plots the training quantities, the loss terms, the robustness
``sqrt(L) / |u - w|_A``, the solution, and the cellwise errors under
``experiments/``.

Run with:

    uv run python examples/03_mf_vpinns_patch.py
"""

import time
from collections.abc import Mapping
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
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
    diagonal_riesz_loss,
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


class SquaredTerms(NamedTuple):
    """
    Squared estimator terms of a trial function.

    Attributes
    ----------
    coupled : jax.Array
        ``eta_P^2 = r^T P^{-1} r``, shape ``()``.
    decoupled : jax.Array
        ``eta_D^2 = r^T D^{-1} r``, shape ``()``.
    residual : jax.Array
        ``sum_i eta_R,i^2``, shape ``()``.
    """

    coupled: jax.Array
    decoupled: jax.Array
    residual: jax.Array


def main(
    cells_per_side: int = 8,
    width: int = 32,
    depth: int = 3,
    num_steps: int = 3000,
    learning_rate: float = 1e-3,
    residual_weight: float = 1.0,
    seed: int = 0,
) -> None:
    """
    Train one network per loss and compare errors and estimator terms.

    Parameters
    ----------
    cells_per_side : int
        Number of grid squares per side; the grid spacing is its inverse.
    width : int
        Hidden-layer width of the network.
    depth : int
        Number of hidden layers.
    num_steps : int
        Number of Adam steps.
    learning_rate : float
        Adam learning rate.
    residual_weight : float
        Weight ``lambda`` of the residual terms in the estimator loss.
    seed : int
        Seed of the initial network weights, shared by all losses.
    """
    family = create_linf_grid_family(
        np.ones((cells_per_side, cells_per_side), dtype=bool),
        spacing=1.0 / cells_per_side,
    )
    problem = create_poisson_sin_sin_problem(family.mesh)
    space = create_patch_test_space(family, problem)
    interior = family.interior_patches
    gram = restrict_to_free(space.gram, interior)
    factor = jnp.linalg.cholesky(gram)
    diagonal = jnp.diagonal(gram)
    points = space.basis.physical_points
    distance = ExactDistanceFunction("unit_square")
    l2_norm = compute_solution_norm(space.basis, problem, l2_error_functional)
    energy_norm = compute_solution_norm(space.basis, problem, energy_error_functional)

    def squared_terms(model: eqx.Module) -> SquaredTerms:
        """
        Squared estimator terms of the constrained network.

        Parameters
        ----------
        model : eqx.Module
            Free network.

        Returns
        -------
        terms : SquaredTerms
            ``eta_P^2``, ``eta_D^2``, and ``sum_i eta_R,i^2``.
        """
        trial_network = ConstrainedAnsatz(model, distance)
        trial = evaluate_network(trial_network, points)
        residual = jnp.take(compute_patch_residual(space, trial), interior, axis=-2)
        flux_divergence = evaluate_network_flux_divergence(
            trial_network, problem.diffusion, points
        )
        return SquaredTerms(
            coupled=riesz_loss(residual, factor),
            decoupled=diagonal_riesz_loss(residual, diagonal),
            residual=jnp.sum(
                compute_patch_residual_terms(space, trial, flux_divergence)
            ),
        )

    def loss_terms(model: eqx.Module, loss_name: str) -> Mapping[str, jax.Array]:
        """
        Named terms of one of the three losses.

        Parameters
        ----------
        model : eqx.Module
            Free network.
        loss_name : str
            ``"decoupled"``, ``"coupled"``, or ``"estimator"``.

        Returns
        -------
        terms : Mapping[str, jax.Array]
            The terms whose sum is the loss.
        """
        terms = squared_terms(model)
        if loss_name == "decoupled":
            return {r"$\eta_D^2$": terms.decoupled}
        if loss_name == "coupled":
            return {r"$\eta_P^2$": terms.coupled}
        return {
            r"$\eta_P^2$": terms.coupled,
            r"$\lambda \sum_i \eta_{R,i}^2$": residual_weight * terms.residual,
        }

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

    output_directory = create_timestamped_output_directory(
        "experiments", "03_mf_vpinns_patch"
    )

    print(
        f"MF-VPINN, Poisson sin-sin, l^inf patches on a "
        f"{cells_per_side}x{cells_per_side} grid: {family.number_of_patches} "
        f"patches, {interior.shape[-1]} test functions, covering bounds "
        f"(c_U2, c_U1) = {family.covering_bounds}."
    )
    for loss_name in ("decoupled", "coupled", "estimator"):

        def run_terms(
            model: eqx.Module, loss_name: str = loss_name
        ) -> Mapping[str, jax.Array]:
            """
            Named terms of the loss of the current run.

            Parameters
            ----------
            model : eqx.Module
                Free network.
            loss_name : str
                Name of the loss of the current run.

            Returns
            -------
            terms : Mapping[str, jax.Array]
                The terms whose sum is the loss.
            """
            return loss_terms(model, loss_name)

        def loss_fn(model: eqx.Module, run_terms=run_terms) -> jax.Array:
            """
            Sum of the loss terms of the current run.

            Parameters
            ----------
            model : eqx.Module
                Free network.
            run_terms : Callable[[eqx.Module], Mapping[str, jax.Array]]
                Terms of the loss of the current run.

            Returns
            -------
            loss : jax.Array
                Loss, shape ``()``.
            """
            return sum(run_terms(model).values())

        network = MultiLayerPerceptron(width=width, depth=depth, key=jr.PRNGKey(seed))
        start = time.perf_counter()
        network, history = train_adam(
            loss_fn,
            network,
            num_steps=num_steps,
            learning_rate=learning_rate,
            monitor=create_training_monitor(run_terms, energy_error, energy_norm),
            monitor_every=max(1, num_steps // 100),
        )
        elapsed = time.perf_counter() - start

        trial_network = ConstrainedAnsatz(network, distance)
        cell_errors = {
            "$L^2$ error": compute_network_error(
                trial_network, space.basis, problem, l2_error_functional, reduce=False
            ),
            "energy error": compute_network_error(
                trial_network,
                space.basis,
                problem,
                energy_error_functional,
                reduce=False,
            ),
        }
        l2 = jnp.sqrt(jnp.sum(cell_errors["$L^2$ error"] ** 2))
        energy = jnp.sqrt(jnp.sum(cell_errors["energy error"] ** 2))
        terms = SquaredTerms(*(jnp.sqrt(term) for term in squared_terms(network)))
        estimator = jnp.sqrt(terms.coupled**2 + residual_weight * terms.residual**2)
        relative_l2 = float(l2 / l2_norm)
        print(
            f"[{loss_name:9s}] {elapsed:6.1f}s | relative L2 {relative_l2:.3e} "
            f"| energy {float(energy):.3e} | eta_P {float(terms.coupled):.3e} "
            f"| eta_D {float(terms.decoupled):.3e} | eta_R {float(terms.residual):.3e} "
            f"| eta_P / energy {float(terms.coupled / energy):.3f} "
            f"| estimator / energy {float(estimator / energy):.3f}"
        )

        figures = {
            "training": plot_training_quantities(history),
            "loss_terms": plot_loss_terms(history),
            "robustness": plot_robustness(history),
            "solution": plot_network_solution(trial_network, family.mesh, problem),
            "cell_errors": plot_cell_errors(family.mesh, cell_errors),
        }
        for name, figure in figures.items():
            save_plot(figure, output_directory / f"{loss_name}_{name}.png")
    print(f"Saved plots to {output_directory}/")


if __name__ == "__main__":
    main()
