"""Convergence study: uniform vs. adaptive refinement, L-shaped re-entrant corner.

Compares the classical benchmark rates (implementation plan Component 7):
uniform refinement on a re-entrant-corner singularity is limited to an
energy-norm rate of O(h^(-1/3)) in DOFs, while Doerfler-adaptive refinement
recovers the optimal O(h^(-1/2)); see ``jax_fem``'s own convergence tests
(``library_analysis.md`` Section 7) for the classical-FEM analogue this
mirrors for MF-VPINNs.
"""

from __future__ import annotations

import jax.random as jr
import matplotlib

matplotlib.use("Agg")

from jax_fem.diagnostics import create_timestamped_output_directory, save_plot
from jax_fem.mesh import create_gmsh_l_shaped_mesh

from vpinns_jax.analysis import plot_convergence_comparison, run_convergence_study
from vpinns_jax.models import ExactDistanceFunction, MultiLayerPerceptron
from vpinns_jax.pde import create_poisson_l_shaped_singular_problem


def main(
    target_h: float = 0.4,
    width: int = 24,
    depth: int = 3,
    num_levels: int = 5,
    adam_steps: int = 1200,
    adam_learning_rate: float = 2e-3,
    dorfler_theta: float = 0.5,
    seed: int = 0,
) -> None:
    mesh = create_gmsh_l_shaped_mesh(target_h)
    distance = ExactDistanceFunction("l_shape")
    exact_solution = create_poisson_l_shaped_singular_problem(mesh).exact_solution

    def lift(x):
        return exact_solution(x[None, None, :])[0, 0, 0]

    def model_factory(key):
        return MultiLayerPerceptron(width=width, depth=depth, key=key)

    key = jr.PRNGKey(seed)
    print(f"Convergence study, L-shaped re-entrant corner, {num_levels} levels each.")

    print("-- uniform refinement --")
    uniform = run_convergence_study(
        mesh,
        create_poisson_l_shaped_singular_problem,
        model_factory,
        distance,
        num_levels=num_levels,
        adaptive=False,
        key=key,
        lift=lift,
        adam_steps=adam_steps,
        adam_learning_rate=adam_learning_rate,
    )
    for record in uniform:
        print(
            f"  level {record.level}: dofs={record.number_of_dofs:5d} "
            f"energy_error={record.energy_error:.4e} eta={record.global_estimator:.4e} "
            f"efficiency={record.efficiency_index:.3f}"
        )

    print("-- adaptive refinement --")
    adaptive = run_convergence_study(
        mesh,
        create_poisson_l_shaped_singular_problem,
        model_factory,
        distance,
        num_levels=num_levels,
        adaptive=True,
        key=key,
        lift=lift,
        adam_steps=adam_steps,
        adam_learning_rate=adam_learning_rate,
        dorfler_theta=dorfler_theta,
    )
    for record in adaptive:
        print(
            f"  level {record.level}: dofs={record.number_of_dofs:5d} "
            f"energy_error={record.energy_error:.4e} eta={record.global_estimator:.4e} "
            f"efficiency={record.efficiency_index:.3f}"
        )

    figure = plot_convergence_comparison(uniform, adaptive, error_attribute="energy_error")
    output_dir = create_timestamped_output_directory("experiments", "05_convergence_study")
    save_plot(figure, output_dir / "convergence.png")
    print(f"Saved plots to {output_dir}/")


if __name__ == "__main__":
    main()
