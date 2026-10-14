"""Interpolated VPINN on a discrete fracture network.

Two orthogonal fractures, ``F1 = (-1, 1) x (0, 1) x {0}`` and
``F2 = {0} x (0, 1) x (-1, 1)``, cross along the trace ``x = z = 0``. The head
has a kink at the trace, and the fractures exchange the fluxes ``+1/3`` (out
of ``F1``) and ``-1/3``. The mesh shares the trace vertices, so the P1
interpolant of a network on ``R^3`` is continuous across the trace and may
kink there; the test space is the conforming P1 space on the embedded cells,
with the per-fracture tangential transmissivity as diffusion.

One network is trained with the Euclidean loss and one with the Riesz loss.
Each is compared with the Galerkin solution, the best the trial can reach,
through its energy distance and the exchanged fluxes. The training plots, the
cellwise L2 and energy errors on every fracture, and the jax-fem fracture and
trace-flux plots are saved under ``experiments/``.

Run with:

    uv run python examples/06_dfn_interpolated.py
"""

import time
from collections.abc import Mapping

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax_fem.constraints import condense_dirichlet_system, expand_condensed_solution
from jax_fem.diagnostics import (
    compute_energy_error,
    compute_l2_error,
    create_timestamped_output_directory,
    energy_error_functional,
    evaluate_trace_fluxes,
    plot_fracture_solutions,
    plot_trace_fluxes,
    save_plot,
)
from jax_fem.element import LagrangeTriangleP1
from jax_fem.function import FiniteElementFunction
from jax_fem.mesh import create_dfn_two_fractures_mesh
from jax_fem.problem import (
    create_dfn_two_fractures_problem,
    evaluate_elliptic_coefficients,
    evaluate_fracture_transmissivity,
)
from jax_fem.reference_cell import ReferenceTriangle
from jax_fem.space import (
    CellBasis,
    create_embedded_cell_basis,
    create_finite_element_space,
)

from vpinns_jax.analysis import (
    compute_solution_norm,
    create_training_monitor,
    plot_cell_errors,
    plot_loss_terms,
    plot_robustness,
    plot_training_quantities,
)
from vpinns_jax.losses import (
    compute_interpolated_fe_residual,
    euclidean_loss,
    restrict_to_free,
    riesz_loss,
)
from vpinns_jax.models import MultiLayerPerceptron, interpolate_network
from vpinns_jax.solvers import train_adam
from vpinns_jax.test_spaces import FETestSpace, create_fe_test_space


def _galerkin_solution(fe: FETestSpace) -> FiniteElementFunction:
    """
    Galerkin solution of the test space's problem.

    Parameters
    ----------
    fe : FETestSpace
        Test space with its stiffness matrix, load vector, and Dirichlet data.

    Returns
    -------
    solution : FiniteElementFunction
        The finite element solution.
    """
    system = condense_dirichlet_system(
        fe.stiffness, fe.load, fe.dirichlet_dofs, fe.dirichlet_values
    )
    free_values = jnp.linalg.solve(system.matrix.todense(), system.vector)
    return FiniteElementFunction(
        space=fe.basis.space,
        dof_values=expand_condensed_solution(free_values, system),
    )


def _trace_exchange(
    solution: FiniteElementFunction, basis: CellBasis, transmissivity: jax.Array
) -> np.ndarray:
    """
    Total outflow of both fractures through every trace.

    Parameters
    ----------
    solution : FiniteElementFunction
        P1 head on the fracture network.
    basis : CellBasis
        Embedded basis of the solution's space.
    transmissivity : jax.Array
        Per-cell transmissivity, shape ``(K, 1, 3, 3)``.

    Returns
    -------
    totals : np.ndarray
        Outflow of the two fractures of each trace, in the order of
        ``mesh.traces_to_fractures``, shape ``(N_traces, 2)``.

    Notes
    -----
    The P1 outflows are constant on each trace facet, so the trapezoidal rule
    over the facet end points is exact.
    """
    mesh = solution.space.mesh
    end_points = jnp.expand_dims(jnp.array([0.0, 1.0]), axis=(0, -2, -1))
    fluxes = evaluate_trace_fluxes(solution, basis, transmissivity, end_points)
    facet_lengths = np.abs(np.diff(np.asarray(fluxes.arc_lengths), axis=-3))
    facet_totals = 0.5 * np.sum(np.asarray(fluxes.outflows), axis=-3, keepdims=True)
    totals = np.zeros((mesh.traces_to_fractures.shape[0], 2))
    np.add.at(
        totals,
        np.asarray(mesh.trace_facets_to_traces),
        (facet_lengths * facet_totals)[:, 0, 0, :],
    )
    return totals


def main(
    cells_per_unit_length: int = 8,
    width: int = 32,
    depth: int = 3,
    num_steps: int = 20000,
    learning_rate: float = 1e-3,
    seed: int = 0,
) -> None:
    """
    Train one network per loss and compare it with the Galerkin solution.

    Parameters
    ----------
    cells_per_unit_length : int
        Mesh resolution of the fractures.
    width : int
        Hidden-layer width of the network.
    depth : int
        Number of hidden layers.
    num_steps : int
        Number of Adam steps.
    learning_rate : float
        Adam learning rate.
    seed : int
        Seed of the initial network weights, shared by both losses.
    """
    mesh = create_dfn_two_fractures_mesh(cells_per_unit_length)
    problem = create_dfn_two_fractures_problem(mesh)
    element = LagrangeTriangleP1(reference_cell=ReferenceTriangle())
    space = create_finite_element_space(mesh, element)
    basis = create_embedded_cell_basis(
        space, element.reference_cell.create_quadrature(4)
    )
    transmissivity = evaluate_fracture_transmissivity(mesh, jnp.ones(2))
    coefficients = evaluate_elliptic_coefficients(
        problem, basis.physical_points
    )._replace(diffusion=transmissivity)
    fe = create_fe_test_space(basis, coefficients, problem)
    factor = jnp.linalg.cholesky(restrict_to_free(fe.gram, fe.free_dofs))
    energy_norm = compute_solution_norm(basis, problem, energy_error_functional)

    galerkin = _galerkin_solution(fe)
    print(
        f"DFN with {mesh.fracture_tags.shape[0]} fractures, {space.number_of_dofs} "
        f"DOFs ({fe.free_dofs.shape[-1]} test functions). Galerkin solution: "
        f"L2 error {float(compute_l2_error(galerkin, basis, problem)):.3e}, energy "
        f"error {float(compute_energy_error(galerkin, basis, problem)):.3e}, trace "
        f"exchange {_trace_exchange(galerkin, basis, transmissivity)[0]} "
        "(exact +-1/3)."
    )

    def energy_error(model: eqx.Module) -> jax.Array:
        """
        Energy error of the interpolated network.

        Parameters
        ----------
        model : eqx.Module
            Network on ``R^3``.

        Returns
        -------
        error : jax.Array
            ``|u - w|_A``, shape ``()``.
        """
        return compute_energy_error(interpolate_network(model, fe), basis, problem)

    output_directory = create_timestamped_output_directory(
        "experiments", "06_dfn_interpolated"
    )
    for loss_name in ("euclidean", "riesz"):

        def loss_terms(
            model: eqx.Module, loss_name: str = loss_name
        ) -> Mapping[str, jax.Array]:
            """
            The single term of the loss of the interpolated network.

            Parameters
            ----------
            model : eqx.Module
                Network on ``R^3``.
            loss_name : str
                ``"euclidean"`` or ``"riesz"``.

            Returns
            -------
            terms : Mapping[str, jax.Array]
                The loss under its name.
            """
            dof_values = interpolate_network(model, fe).dof_values
            residual = jnp.take(
                compute_interpolated_fe_residual(fe, dof_values), fe.free_dofs, axis=-2
            )
            if loss_name == "riesz":
                return {r"$r^T G^{-1} r$": riesz_loss(residual, factor)}
            return {r"$r^T r$": euclidean_loss(residual)}

        def loss_fn(model: eqx.Module, loss_terms=loss_terms) -> jax.Array:
            """
            Sum of the loss terms.

            Parameters
            ----------
            model : eqx.Module
                Network on ``R^3``.
            loss_terms : Callable[[eqx.Module], Mapping[str, jax.Array]]
                Terms of the loss of the current run.

            Returns
            -------
            loss : jax.Array
                Loss, shape ``()``.
            """
            return sum(loss_terms(model).values())

        network = MultiLayerPerceptron(
            input_dimension=3, width=width, depth=depth, key=jr.PRNGKey(seed)
        )
        start = time.perf_counter()
        network, history = train_adam(
            loss_fn,
            network,
            num_steps=num_steps,
            learning_rate=learning_rate,
            monitor=create_training_monitor(loss_terms, energy_error, energy_norm),
            monitor_every=max(1, num_steps // 100),
        )
        elapsed = time.perf_counter() - start

        solution = interpolate_network(network, fe)
        difference = solution.dof_values - galerkin.dof_values
        galerkin_distance = jnp.sqrt(jnp.sum(difference * (fe.stiffness @ difference)))
        print(
            f"[{loss_name:9s}] {elapsed:.1f}s | L2 error "
            f"{float(compute_l2_error(solution, basis, problem)):.3e} | energy error "
            f"{float(compute_energy_error(solution, basis, problem)):.3e} | energy "
            f"distance to Galerkin {float(galerkin_distance):.3e} | trace exchange "
            f"{_trace_exchange(solution, basis, transmissivity)[0]}"
        )

        figures = {
            "training": plot_training_quantities(history),
            "loss_terms": plot_loss_terms(history),
            "robustness": plot_robustness(history),
            "cell_errors": plot_cell_errors(
                mesh,
                {
                    "$L^2$ error": compute_l2_error(
                        solution, basis, problem, reduce=False
                    ),
                    "energy error": compute_energy_error(
                        solution, basis, problem, reduce=False
                    ),
                },
            ),
            **plot_fracture_solutions(solution),
            **plot_trace_fluxes(solution, basis, transmissivity),
        }
        for name, figure in figures.items():
            save_plot(figure, output_directory / f"{loss_name}_{name}.png")
    print(f"Saved plots to {output_directory}/")


if __name__ == "__main__":
    main()
