"""Interpolated VPINN for the Poisson problem with solution sin(pi x) sin(pi y).

The trial function is the P1 interpolant of a network, with the Dirichlet
values imposed at the boundary DOFs. Since it is a finite element function,
its residual is ``stiffness @ w - load``, restricted to the interior DOFs. The
network is trained with the Euclidean loss; the training quantities, the
robustness ``sqrt(L) / |u - w|_A``, and the cellwise L2 and energy errors of the
interpolant on the mesh are plotted under ``experiments/``.

Run with:

    uv run python examples/02_vpinns_interpolated.py
"""

import time
from collections.abc import Mapping

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jax_fem.diagnostics import (
    compute_energy_error,
    compute_h1_seminorm_error,
    compute_l2_error,
    create_timestamped_output_directory,
    energy_error_functional,
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
    compute_solution_norm,
    create_training_monitor,
    plot_cell_errors,
    plot_loss_terms,
    plot_robustness,
    plot_training_quantities,
)
from vpinns_jax.losses import compute_interpolated_fe_residual, euclidean_loss
from vpinns_jax.models import MultiLayerPerceptron, interpolate_network
from vpinns_jax.solvers import train_adam
from vpinns_jax.test_spaces import create_fe_test_space


def main(
    cells_per_side: int = 16,
    width: int = 32,
    depth: int = 3,
    num_steps: int = 10000,
    learning_rate: float = 1e-3,
    seed: int = 0,
) -> None:
    """
    Train the interpolated VPINN, report the errors of the interpolant, and plot.

    Parameters
    ----------
    cells_per_side : int
        Mesh resolution of the unit square.
    width : int
        Hidden-layer width of the network.
    depth : int
        Number of hidden layers.
    num_steps : int
        Number of Adam steps.
    learning_rate : float
        Adam learning rate.
    seed : int
        Seed of the initial network weights.
    """
    mesh = create_gmsh_unit_square_mesh(cells_per_side)
    problem = create_poisson_sin_sin_problem(mesh)
    element = LagrangeTriangleP1(reference_cell=ReferenceTriangle())
    space = create_finite_element_space(mesh, element)
    basis = create_cell_basis(space, element.reference_cell.create_quadrature(4))
    coefficients = evaluate_elliptic_coefficients(problem, basis.physical_points)
    fe = create_fe_test_space(basis, coefficients, problem)

    def loss_terms(model: eqx.Module) -> Mapping[str, jax.Array]:
        """
        The single term of the Euclidean loss of the interpolated network.

        Parameters
        ----------
        model : eqx.Module
            Network.

        Returns
        -------
        terms : Mapping[str, jax.Array]
            The loss under its name.
        """
        dof_values = interpolate_network(model, fe).dof_values
        residual = compute_interpolated_fe_residual(fe, dof_values)
        return {r"$r^T r$": euclidean_loss(jnp.take(residual, fe.free_dofs, axis=-2))}

    def loss_fn(model: eqx.Module) -> jax.Array:
        """
        Euclidean loss of the interpolated network.

        Parameters
        ----------
        model : eqx.Module
            Network.

        Returns
        -------
        loss : jax.Array
            Loss, shape ``()``.
        """
        return sum(loss_terms(model).values())

    def energy_error(model: eqx.Module) -> jax.Array:
        """
        Energy error of the interpolated network.

        Parameters
        ----------
        model : eqx.Module
            Network.

        Returns
        -------
        error : jax.Array
            ``|u - w|_A``, shape ``()``.
        """
        return compute_energy_error(interpolate_network(model, fe), basis, problem)

    monitor = create_training_monitor(
        loss_terms,
        energy_error,
        compute_solution_norm(basis, problem, energy_error_functional),
    )
    print(
        f"Interpolated VPINN, Poisson sin-sin, {cells_per_side}x{cells_per_side} mesh "
        f"({space.number_of_dofs} DOFs, {fe.free_dofs.shape[-1]} test functions)."
    )
    network = MultiLayerPerceptron(width=width, depth=depth, key=jr.PRNGKey(seed))
    start = time.perf_counter()
    network, history = train_adam(
        loss_fn,
        network,
        num_steps=num_steps,
        learning_rate=learning_rate,
        log_every=max(1, num_steps // 10),
        monitor=monitor,
        monitor_every=max(1, num_steps // 100),
    )
    elapsed = time.perf_counter() - start
    print(f"Trained {history.loss.shape[-1]} Adam steps in {elapsed:.2f}s.")

    solution = interpolate_network(network, fe)
    l2_norm = compute_solution_norm(basis, problem, l2_error_functional)
    l2 = compute_l2_error(solution, basis, problem)
    h1 = compute_h1_seminorm_error(solution, basis, problem)
    energy = compute_energy_error(solution, basis, problem)
    print(f"Relative L2 error : {float(l2 / l2_norm):.6e}")
    print(f"H1 seminorm error : {float(h1):.6e}")
    print(f"Energy norm error : {float(energy):.6e}")

    output_directory = create_timestamped_output_directory(
        "experiments", "02_vpinns_interpolated"
    )
    cell_errors = {
        "$L^2$ error": compute_l2_error(solution, basis, problem, reduce=False),
        "energy error": compute_energy_error(solution, basis, problem, reduce=False),
    }
    figures = {
        "training": plot_training_quantities(history),
        "loss_terms": plot_loss_terms(history),
        "robustness": plot_robustness(history),
        "cell_errors": plot_cell_errors(mesh, cell_errors),
    }
    for name, figure in figures.items():
        save_plot(figure, output_directory / f"{name}.png")
    print(f"Saved plots to {output_directory}/")


if __name__ == "__main__":
    main()
