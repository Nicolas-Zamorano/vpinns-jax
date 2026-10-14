"""Mesh-free VPINN with the Shepard-normalised vertex family of l^inf cones.

The patches are the squares ``|x - z|_inf < s`` at the grid points of the unit
square, cut to the domain, with weights ``1 - |x - z|_inf / s``. Their sum
``S`` lies between one and two, so the test functions are the normalised
weights ``psi_i = phi_i / S`` of the interior patches. Every patch is
integrated on its own triangles with the degree-2 rule, evaluating ``S`` from
the weights of its neighbours; no global mesh is built. The trial function is a
network with the homogeneous Dirichlet condition imposed by the ansatz
``w = B u_theta``, for the Poisson problem with solution sin(pi x) sin(pi y).
Four losses are compared, every combination of a dual norm of the residual
with or without the residual terms:

- decoupled: ``eta_D^2 = r^T D^{-1} r``, with ``D`` the diagonal of the energy
  Gram matrix ``P`` of the test functions;
- coupled: ``eta_P^2 = r^T P^{-1} r``, the dual norm of the residual over the
  patch space;
- decoupled_estimator: ``eta_D^2 + lambda sum_i eta_R,i^2``;
- coupled_estimator: ``eta_P^2 + lambda sum_i eta_R,i^2``;

where the residual terms are the weighted strong residuals
``eta_R,i = h_i ||phi_i^(1/2) (f + lap w)||``.

For every trained network the script prints the errors, the three estimator
terms, and the ratios of ``eta_P`` and of both estimators,
``(eta_P^2 + lambda eta_R^2)^(1/2)`` and ``(eta_D^2 + lambda eta_R^2)^(1/2)``,
to the energy error, and plots the training quantities, the loss terms, the
robustness ``sqrt(L) / |u - w|_A``, and the exact and network solutions with
their pointwise error, as surfaces on a uniform grid, under ``experiments/``.
The errors are measured on a finite element mesh of the unit square with the
degree-4 quadrature, independent of the patches.

Run with:

    uv run python examples/09_mf_vpinns_linf_vertex.py
"""

import time
from collections.abc import Mapping
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jax_fem.diagnostics import (
    create_timestamped_output_directory,
    energy_error_functional,
    l2_error_functional,
    save_plot,
)
from jax_fem.element import LagrangeTriangleP1
from jax_fem.mesh import create_gmsh_unit_square_mesh
from jax_fem.problem import create_poisson_sin_sin_problem
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
    compute_meshfree_patch_residual,
    compute_meshfree_patch_residual_terms,
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
from vpinns_jax.test_spaces import (
    create_linf_vertex_family,
    create_meshfree_patch_test_space,
)


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
    num_steps: int = 10000,
    learning_rate: float = 1e-3,
    residual_weight: float = 1.0,
    seed: int = 0,
    quadrature_degree: int = 2,
    error_cells_per_side: int = 64,
    error_quadrature_degree: int = 4,
) -> None:
    """
    Train one network per loss and compare errors and estimator terms.

    Parameters
    ----------
    cells_per_side : int
        Number of grid cells per side; the grid spacing is its inverse.
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
    quadrature_degree : int
        Exactness degree of the rule on the triangles of every patch.
    error_cells_per_side : int
        Resolution of the mesh on which the errors are measured.
    error_quadrature_degree : int
        Exactness degree of the rule on the cells of that mesh.
    """
    element = LagrangeTriangleP1(reference_cell=ReferenceTriangle())
    error_mesh = create_gmsh_unit_square_mesh(error_cells_per_side)
    error_basis = create_cell_basis(
        create_finite_element_space(error_mesh, element),
        element.reference_cell.create_quadrature(error_quadrature_degree),
    )
    # The mesh only defines the problem's Dirichlet data; the patch space uses
    # its coefficient functions.
    problem = create_poisson_sin_sin_problem(error_mesh)
    l2_norm = compute_solution_norm(error_basis, problem, l2_error_functional)
    energy_norm = compute_solution_norm(error_basis, problem, energy_error_functional)

    family = create_linf_vertex_family(cells_per_side)
    space = create_meshfree_patch_test_space(
        family, problem, quadrature_degree=quadrature_degree, normalise=True
    )
    interior = family.interior_patches
    gram = restrict_to_free(space.gram, interior)
    factor = jnp.linalg.cholesky(gram)
    diagonal = jnp.diagonal(gram)
    points = space.points
    distance = ExactDistanceFunction("unit_square")

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
        residual = jnp.take(
            compute_meshfree_patch_residual(space, trial), interior, axis=-2
        )
        flux_divergence = evaluate_network_flux_divergence(
            trial_network, problem.diffusion, points
        )
        return SquaredTerms(
            coupled=riesz_loss(residual, factor),
            decoupled=diagonal_riesz_loss(residual, diagonal),
            residual=jnp.sum(
                compute_meshfree_patch_residual_terms(space, trial, flux_divergence)
            ),
        )

    def loss_terms(model: eqx.Module, loss_name: str) -> Mapping[str, jax.Array]:
        """
        Named terms of one of the four losses.

        Parameters
        ----------
        model : eqx.Module
            Free network.
        loss_name : str
            ``"decoupled"``, ``"coupled"``, ``"decoupled_estimator"``, or
            ``"coupled_estimator"``.

        Returns
        -------
        terms : Mapping[str, jax.Array]
            The terms whose sum is the loss.
        """
        terms = squared_terms(model)
        if loss_name.startswith("decoupled"):
            dual_norm = {r"$\eta_D^2$": terms.decoupled}
        else:
            dual_norm = {r"$\eta_P^2$": terms.coupled}
        if loss_name.endswith("estimator"):
            residual = residual_weight * terms.residual
            return dual_norm | {r"$\lambda \sum_i \eta_{R,i}^2$": residual}
        return dual_norm

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
            error_basis,
            problem,
            energy_error_functional,
        )

    output_directory = create_timestamped_output_directory(
        "experiments", "09_mf_vpinns_linf_vertex"
    )

    print(
        f"MF-VPINN, Poisson sin-sin, Shepard-normalised l^inf patches on a "
        f"{cells_per_side}x{cells_per_side} grid: {family.number_of_patches} "
        f"patches, {interior.shape[-1]} test functions, covering bounds "
        f"(c_U2, c_U1) = {space.covering_bounds}."
    )
    for loss_name in (
        "decoupled",
        "coupled",
        "decoupled_estimator",
        "coupled_estimator",
    ):

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
            monitor_every=1,
        )
        elapsed = time.perf_counter() - start

        trial_network = ConstrainedAnsatz(network, distance)
        l2 = compute_network_error(
            trial_network, error_basis, problem, l2_error_functional
        )
        energy = compute_network_error(
            trial_network, error_basis, problem, energy_error_functional
        )
        terms = SquaredTerms(*(jnp.sqrt(term) for term in squared_terms(network)))
        residual_estimate = residual_weight * terms.residual**2
        coupled_estimator = jnp.sqrt(terms.coupled**2 + residual_estimate)
        decoupled_estimator = jnp.sqrt(terms.decoupled**2 + residual_estimate)
        relative_l2 = float(l2 / l2_norm)
        print(
            f"[{loss_name:19s}] {elapsed:6.1f}s | relative L2 {relative_l2:.3e} "
            f"| energy {float(energy):.3e} | eta_P {float(terms.coupled):.3e} "
            f"| eta_D {float(terms.decoupled):.3e} | eta_R {float(terms.residual):.3e} "
            f"| eta_P / energy {float(terms.coupled / energy):.3f} "
            f"| coupled estimator / energy {float(coupled_estimator / energy):.3f} "
            f"| decoupled estimator / energy {float(decoupled_estimator / energy):.3f}"
        )

        figures = {
            "training": plot_training_quantities(history),
            "loss_terms": plot_loss_terms(history),
            "robustness": plot_robustness(history),
            "solution": plot_network_solution(trial_network, error_mesh, problem),
        }
        for name, figure in figures.items():
            save_plot(figure, output_directory / f"{loss_name}_{name}.png")
    print(f"Saved plots to {output_directory}/")


if __name__ == "__main__":
    main()
