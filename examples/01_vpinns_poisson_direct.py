"""Direct VPINN for the Poisson problem with solution sin(pi x) sin(pi y).

The trial function is a network with the homogeneous Dirichlet condition
imposed by the ansatz ``w = B u_theta``, ``B = x (1 - x) y (1 - y)``, tested
against the P1 basis functions of the interior vertices. One network is
trained with the Euclidean loss ``r^T r`` and one with the Riesz loss
``r^T G^{-1} r``, where ``G`` is the energy Gram matrix of the test functions,
and their errors are compared. For each loss the training quantities, the
robustness ``sqrt(L) / |u - w|_A``, and the exact and network solutions with
their pointwise error, as surfaces on a uniform grid, are plotted under
``experiments/``.

Run with:

    uv run python examples/01_vpinns_poisson_direct.py
"""

import time
from collections.abc import Callable, Mapping

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jax_fem.diagnostics import (
    create_timestamped_output_directory,
    energy_error_functional,
    h1_seminorm_error_functional,
    l2_error_functional,
    save_plot,
)
from jax_fem.element import LagrangeTriangleP1
from jax_fem.mesh import create_gmsh_unit_square_mesh
from jax_fem.problem import (
    create_poisson_sin_sin_problem,
    evaluate_elliptic_coefficients,
)
from jax_fem.reference_cell import ReferenceTriangle
from jax_fem.space import create_cell_basis, create_finite_element_space

from vpinns_jax.analysis import (
    compute_network_error,
    compute_solution_norm,
    create_training_monitor,
    plot_loss_terms,
    plot_network_solution,
    plot_robustness,
    plot_training_quantities,
)
from vpinns_jax.losses import (
    compute_fe_residual,
    euclidean_loss,
    restrict_to_free,
    riesz_loss,
)
from vpinns_jax.models import (
    ConstrainedAnsatz,
    ExactDistanceFunction,
    MultiLayerPerceptron,
    evaluate_network,
)
from vpinns_jax.solvers import Monitor, TrainingHistory, train_adam, train_lbfgs
from vpinns_jax.test_spaces import create_fe_test_space


def _train(
    loss_fn: Callable[[eqx.Module], jax.Array],
    network: eqx.Module,
    optimizer: str,
    num_steps: int,
    learning_rate: float,
    monitor: Monitor,
) -> tuple[eqx.Module, TrainingHistory]:
    """
    Train a network with the chosen optimizer.

    Parameters
    ----------
    loss_fn : Callable[[eqx.Module], jax.Array]
        Scalar loss of the network.
    network : eqx.Module
        Initial network.
    optimizer : str
        ``"adam"`` or ``"lbfgs"``.
    num_steps : int
        Number of steps.
    learning_rate : float
        Adam learning rate; unused by L-BFGS.
    monitor : Monitor
        Training monitor, recorded about a hundred times.

    Returns
    -------
    network : eqx.Module
        Trained network.
    history : TrainingHistory
        Loss of every step and the monitor records.
    """
    options = {
        "num_steps": num_steps,
        "log_every": max(1, num_steps // 5),
        "monitor": monitor,
        "monitor_every": max(1, num_steps // 100),
    }
    if optimizer == "adam":
        return train_adam(loss_fn, network, learning_rate=learning_rate, **options)
    if optimizer == "lbfgs":
        return train_lbfgs(loss_fn, network, **options)
    raise ValueError(f"optimizer must be 'adam' or 'lbfgs', got {optimizer!r}.")


def main(
    cells_per_side: int = 16,
    width: int = 32,
    depth: int = 3,
    optimizer: str = "adam",
    num_steps: int = 3000,
    learning_rate: float = 1e-3,
    seed: int = 0,
) -> None:
    """
    Train one network per loss, compare their errors, and plot them.

    Parameters
    ----------
    cells_per_side : int
        Mesh resolution of the unit square.
    width : int
        Hidden-layer width of the network.
    depth : int
        Number of hidden layers.
    optimizer : str
        ``"adam"`` or ``"lbfgs"``.
    num_steps : int
        Number of training steps.
    learning_rate : float
        Adam learning rate.
    seed : int
        Seed of the initial network weights, shared by both losses.
    """
    mesh = create_gmsh_unit_square_mesh(cells_per_side)
    problem = create_poisson_sin_sin_problem(mesh)
    element = LagrangeTriangleP1(reference_cell=ReferenceTriangle())
    space = create_finite_element_space(mesh, element)
    basis = create_cell_basis(space, element.reference_cell.create_quadrature(4))
    coefficients = evaluate_elliptic_coefficients(problem, basis.physical_points)
    fe = create_fe_test_space(basis, coefficients, problem)
    factor = jnp.linalg.cholesky(restrict_to_free(fe.gram, fe.free_dofs))
    distance = ExactDistanceFunction("unit_square")
    l2_norm = compute_solution_norm(basis, problem, l2_error_functional)
    energy_norm = compute_solution_norm(basis, problem, energy_error_functional)

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
            ConstrainedAnsatz(model, distance), basis, problem, energy_error_functional
        )

    output_directory = create_timestamped_output_directory(
        "experiments", "01_vpinns_poisson_direct"
    )
    print(
        f"Direct VPINN, Poisson sin-sin, {cells_per_side}x{cells_per_side} mesh "
        f"({space.number_of_dofs} DOFs, {fe.free_dofs.shape[-1]} test functions), "
        f"{optimizer}."
    )
    for loss_name in ("euclidean", "riesz"):

        def loss_terms(
            model: eqx.Module, loss_name: str = loss_name
        ) -> Mapping[str, jax.Array]:
            """
            The single term of the loss of the constrained network.

            Parameters
            ----------
            model : eqx.Module
                Free network.
            loss_name : str
                ``"euclidean"`` or ``"riesz"``.

            Returns
            -------
            terms : Mapping[str, jax.Array]
                The loss under its name.
            """
            trial = evaluate_network(
                ConstrainedAnsatz(model, distance), basis.physical_points
            )
            residual = jnp.take(compute_fe_residual(fe, trial), fe.free_dofs, axis=-2)
            if loss_name == "riesz":
                return {r"$r^T G^{-1} r$": riesz_loss(residual, factor)}
            return {r"$r^T r$": euclidean_loss(residual)}

        def loss_fn(model: eqx.Module, loss_terms=loss_terms) -> jax.Array:
            """
            Sum of the loss terms.

            Parameters
            ----------
            model : eqx.Module
                Free network.
            loss_terms : Callable[[eqx.Module], Mapping[str, jax.Array]]
                Terms of the loss of the current run.

            Returns
            -------
            loss : jax.Array
                Loss, shape ``()``.
            """
            return sum(loss_terms(model).values())

        monitor = create_training_monitor(loss_terms, energy_error, energy_norm)
        network = MultiLayerPerceptron(width=width, depth=depth, key=jr.PRNGKey(seed))
        start = time.perf_counter()
        network, history = _train(
            loss_fn, network, optimizer, num_steps, learning_rate, monitor
        )
        elapsed = time.perf_counter() - start

        trial = ConstrainedAnsatz(network, distance)
        l2 = compute_network_error(trial, basis, problem, l2_error_functional)
        h1 = compute_network_error(trial, basis, problem, h1_seminorm_error_functional)
        energy = compute_network_error(trial, basis, problem, energy_error_functional)
        print(
            f"[{loss_name:9s}] {elapsed:.2f}s | relative L2 {float(l2 / l2_norm):.4e} "
            f"| H1 seminorm {float(h1):.4e} | energy {float(energy):.4e} "
            f"| robustness {float(history.monitored.robustness[-1]):.3f}"
        )

        figures = {
            "training": plot_training_quantities(history),
            "loss_terms": plot_loss_terms(history),
            "robustness": plot_robustness(history),
            "solution": plot_network_solution(trial, mesh, problem),
        }
        for name, figure in figures.items():
            save_plot(figure, output_directory / f"{loss_name}_{name}.png")
    print(f"Saved plots to {output_directory}/")


if __name__ == "__main__":
    main()
