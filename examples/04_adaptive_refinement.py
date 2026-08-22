"""Closed-loop adaptive refinement on the L-shaped re-entrant-corner singularity.

SOLVE -> ESTIMATE -> MARK -> REFINE (math reference Section 4.4), driven by
the mesh-free patch a posteriori estimator with the jump-term reduction
(Section 4.3: the direct trial is smooth, so no facet quadrature is needed),
and jax-fem's Rivara LEB engine for REFINE. The network's parameters
warm-start across levels (``solvers.adaptive_loop.run_adaptive_loop``) --
unlike classical FE AMR, the trial function does not depend on the mesh.

Domain: ``[-1,1]^2 \\ [0,1]x[-1,0]``, genuine 270-degree re-entrant corner at
the origin. Solution ``u(r, theta') = r^(2/3) sin(2/3 theta')`` has a
singular gradient there, so an adaptive mesh should visibly cluster cells
near the origin -- this is the standard benchmark for which a uniformly
refined mesh under-performs an adaptively refined one.
"""

from __future__ import annotations

import jax.random as jr
import matplotlib

matplotlib.use("Agg")

from jax_fem.diagnostics import create_timestamped_output_directory, save_plot
from jax_fem.mesh import create_gmsh_l_shaped_mesh

from vpinns_jax.models import ExactDistanceFunction, MultiLayerPerceptron
from vpinns_jax.pde import create_poisson_l_shaped_singular_problem
from vpinns_jax.solvers import run_adaptive_loop


def _plot_mesh(mesh, title: str):
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(5, 5))
    axis.triplot(
        mesh.vertex_coordinates[:, 0, 0],
        mesh.vertex_coordinates[:, 0, 1],
        mesh.cells_to_vertices,
        color="k",
        linewidth=0.4,
    )
    axis.set_aspect("equal")
    axis.set_title(title)
    return figure


def main(
    target_h: float = 0.35,
    width: int = 32,
    depth: int = 3,
    num_levels: int = 6,
    adam_steps: int = 1500,
    adam_learning_rate: float = 2e-3,
    dorfler_theta: float = 0.5,
    max_marked_fraction: float = 0.3,
    seed: int = 0,
) -> None:
    mesh = create_gmsh_l_shaped_mesh(target_h)
    distance = ExactDistanceFunction("l_shape")
    network = MultiLayerPerceptron(width=width, depth=depth, key=jr.PRNGKey(seed))

    # Non-homogeneous Dirichlet data: lift the strong-BC ansatz with the
    # exact solution itself (an admissible H^1 extension of the boundary
    # trace) -- omitting this would silently solve the WRONG (homogeneous)
    # problem, since ConstrainedAnsatz's `lift` defaults to zero. This
    # problem's `exact_solution` is a pure function of coordinates (no mesh
    # dependence), so it is safe to capture once here and reuse across every
    # refinement level, unlike `dirichlet_conditions` (which does reference
    # mesh-specific boundary tags and is rebuilt per level by
    # `problem_factory` inside `run_adaptive_loop`).
    exact_solution = create_poisson_l_shaped_singular_problem(mesh).exact_solution

    def lift(x):
        return exact_solution(x[None, None, :])[0, 0, 0]

    print(f"Adaptive MF-VPINN, L-shaped re-entrant corner, {num_levels} levels, "
          f"initial mesh {mesh.cells_to_vertices.shape[0]} cells.")

    results = run_adaptive_loop(
        mesh,
        create_poisson_l_shaped_singular_problem,
        network,
        distance,
        lift=lift,
        num_levels=num_levels,
        adam_steps=adam_steps,
        adam_learning_rate=adam_learning_rate,
        dorfler_theta=dorfler_theta,
        max_marked_fraction=max_marked_fraction,
        log_every=None,
    )

    for level, result in enumerate(results):
        cells = result.mesh.cells_to_vertices.shape[0]
        print(
            f"level {level}: cells={cells:5d} patches={result.number_of_patches:5d} "
            f"marked={result.number_of_marked_cells:5d} "
            f"final_loss={float(result.history.loss[-1]):.4e} eta={result.global_estimator:.4e}"
        )

    output_dir = create_timestamped_output_directory("experiments", "04_adaptive_refinement")
    save_plot(
        _plot_mesh(results[0].mesh, f"Level 0 ({results[0].mesh.cells_to_vertices.shape[0]} cells)"),
        output_dir / "mesh_level_0.png",
    )
    save_plot(
        _plot_mesh(
            results[-1].mesh, f"Level {len(results) - 1} ({results[-1].mesh.cells_to_vertices.shape[0]} cells)"
        ),
        output_dir / "mesh_final.png",
    )

    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(6, 4))
    cell_counts = [r.mesh.cells_to_vertices.shape[0] for r in results]
    estimators = [r.global_estimator for r in results]
    axis.plot(cell_counts, estimators, "o-")
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("number of cells")
    axis.set_ylabel(r"global estimator $\eta$")
    axis.set_title("Estimator decay under adaptive refinement")
    axis.grid(True, which="both", linestyle=":", linewidth=0.5)
    save_plot(figure, output_dir / "estimator_decay.png")

    print(f"Saved plots to {output_dir}/")


if __name__ == "__main__":
    main()
