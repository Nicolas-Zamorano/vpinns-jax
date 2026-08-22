"""Interpolated VPINN on the Poisson sin(pi x) sin(pi y) problem: w_theta = Pi_h u_theta.

Trial: interpolated, coordinate interpolation with a strong BC (math
reference Section 3 "strong, on the interpolant"; see also
``src/examples/00_poc_interpolated_vpinns_poisson_sin_sin.py``, this
example's original throwaway proof of concept -- this version is the same
idea built entirely from ``vpinns_jax`` library calls).
Test space: FE (V_h), restricted to interior DOFs.
Loss: Euclidean -- w_theta is already a P1 FE function once interpolated, so
the residual is the plain sparse matvec ``stiffness @ w_dofs - load``
(``losses.residual.interpolated_trial_fe_residual``).
"""

from __future__ import annotations

import time

import jax.random as jr
import matplotlib

matplotlib.use("Agg")

from jax_fem.diagnostics import (
    create_timestamped_output_directory,
    plot_fem_error,
    plot_fem_error_3d,
    save_plot,
)
from jax_fem.function import FiniteElementFunction
from jax_fem.mesh import create_gmsh_unit_square_mesh

from vpinns_jax.analysis import (
    compute_energy_error,
    compute_h1_seminorm_error,
    compute_relative_l2_error,
)
from vpinns_jax.losses import euclidean_loss, interpolated_trial_fe_residual
from vpinns_jax.models import (
    MultiLayerPerceptron,
    interpolated_trial_dofs,
    precompute_interpolation_data,
)
from vpinns_jax.pde import create_poisson_sin_sin_problem
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
    mesh = create_gmsh_unit_square_mesh(cells_per_side)
    problem = create_poisson_sin_sin_problem(mesh)
    fe = create_fe_test_space(mesh, problem)
    data = precompute_interpolation_data(fe.space, problem)

    network = MultiLayerPerceptron(width=width, depth=depth, key=jr.PRNGKey(seed))

    def loss_fn(trial_model):
        w_dofs = interpolated_trial_dofs(trial_model, data)
        residual = interpolated_trial_fe_residual(w_dofs, fe)[fe.free_dofs, 0]
        return euclidean_loss(residual)

    print(
        f"Interpolated VPINN, Poisson sin-sin, {cells_per_side}x{cells_per_side} P1 mesh "
        f"({fe.space.number_of_dofs} DOFs, {fe.number_of_test_functions} free)."
    )
    start = time.perf_counter()
    network, history = train_adam(
        loss_fn,
        network,
        num_steps=num_steps,
        learning_rate=learning_rate,
        log_every=num_steps // 10,
    )
    print(f"Trained {num_steps} Adam steps in {time.perf_counter() - start:.2f}s.")

    final_dofs = interpolated_trial_dofs(network, data)
    solution = FiniteElementFunction(space=fe.space, dof_values=final_dofs)

    relative_l2 = float(compute_relative_l2_error(solution, fe.basis, problem))
    h1_seminorm = float(compute_h1_seminorm_error(solution, fe.basis, problem))
    energy = float(compute_energy_error(solution, fe.basis, problem))
    print(f"Relative L2 error : {relative_l2:.6e}")
    print(f"H1 seminorm error : {h1_seminorm:.6e}")
    print(f"Energy norm error : {energy:.6e}")

    import matplotlib.pyplot as plt

    loss_figure, axis = plt.subplots(figsize=(6, 4))
    axis.plot(history.loss)
    axis.set_yscale("log")
    axis.set_xlabel("Adam step")
    axis.set_ylabel("Euclidean loss")
    axis.set_title("Interpolated VPINN training loss")
    axis.grid(True, which="both", linestyle=":", linewidth=0.5)

    output_dir = create_timestamped_output_directory(
        "experiments", problem.name + "_interpolated"
    )
    save_plot(loss_figure, output_dir / "training.png")
    save_plot(
        plot_fem_error_3d(solution, fe.basis, problem, mesh), output_dir / "3d.png"
    )
    save_plot(
        plot_fem_error(solution, fe.basis, problem, mesh), output_dir / "mesh_error.png"
    )
    print(f"Saved plots to {output_dir}/")


if __name__ == "__main__":
    main()
