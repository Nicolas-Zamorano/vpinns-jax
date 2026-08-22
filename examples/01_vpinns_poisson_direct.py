"""Direct VPINN on the Poisson sin(pi x) sin(pi y) problem: strong BCs, FE test space.

Trial: direct, w_theta = u_theta^g = B(x) u_theta(x) (homogeneous BC, math
reference Section 3 "strong, through the network ansatz").
Test space: FE (V_h), restricted to interior DOFs.
Loss: trains one network with the Euclidean loss and a second with the
Riesz (robust) loss, then compares their exact errors -- the point of this
example is exactly that comparison (math reference Section 2.3).
"""

from __future__ import annotations

import time

import jax.random as jr
import matplotlib

matplotlib.use("Agg")

from jax_fem.diagnostics import create_timestamped_output_directory, save_plot
from jax_fem.mesh import create_gmsh_unit_square_mesh

from vpinns_jax.analysis import (
    direct_trial_energy_error,
    direct_trial_h1_seminorm_error,
    direct_trial_relative_l2_error,
)
from vpinns_jax.losses import (
    assemble_energy_gram_matrix,
    direct_trial_fe_residual,
    euclidean_loss,
    restrict_to_free_dofs,
    riesz_loss_global,
)
from vpinns_jax.models import (
    ConstrainedAnsatz,
    ExactDistanceFunction,
    MultiLayerPerceptron,
)
from vpinns_jax.pde import create_poisson_sin_sin_problem
from vpinns_jax.solvers import train_adam
from vpinns_jax.test_spaces import create_fe_test_space


def main(
    cells_per_side: int = 16,
    width: int = 32,
    depth: int = 3,
    num_steps: int = 3000,
    learning_rate: float = 1e-3,
    seed: int = 0,
) -> None:
    mesh = create_gmsh_unit_square_mesh(cells_per_side)
    problem = create_poisson_sin_sin_problem(mesh)
    fe = create_fe_test_space(mesh, problem)
    distance = ExactDistanceFunction("unit_square")

    gram_free = restrict_to_free_dofs(
        assemble_energy_gram_matrix(fe, problem), fe.free_dofs
    )

    print(
        f"Direct VPINN, Poisson sin-sin, {cells_per_side}x{cells_per_side} P1 mesh "
        f"({fe.space.number_of_dofs} DOFs, {fe.number_of_test_functions} free)."
    )

    histories = {}
    trials = {}
    for loss_name in ("euclidean", "riesz"):
        network = MultiLayerPerceptron(width=width, depth=depth, key=jr.PRNGKey(seed))

        def loss_fn(trial_model, loss_name=loss_name):
            trial = ConstrainedAnsatz(trial_model, distance)
            residual = direct_trial_fe_residual(trial, fe)[fe.free_dofs, 0]
            if loss_name == "riesz":
                return riesz_loss_global(residual, gram_free)
            return euclidean_loss(residual)

        start = time.perf_counter()
        network, history = train_adam(
            loss_fn,
            network,
            num_steps=num_steps,
            learning_rate=learning_rate,
            log_every=num_steps // 5,
        )
        elapsed = time.perf_counter() - start
        trial = ConstrainedAnsatz(network, distance)

        relative_l2 = float(direct_trial_relative_l2_error(trial, fe.basis, problem))
        h1_seminorm = float(direct_trial_h1_seminorm_error(trial, fe.basis, problem))
        energy = float(direct_trial_energy_error(trial, fe.basis, problem))
        print(
            f"[{loss_name:9s}] trained in {elapsed:.2f}s | "
            f"relative L2 {relative_l2:.4e} | H1 {h1_seminorm:.4e} | energy {energy:.4e}"
        )
        histories[loss_name] = history
        trials[loss_name] = trial

    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(6, 4))
    for loss_name, history in histories.items():
        axis.plot(history.loss, label=loss_name)
    axis.set_yscale("log")
    axis.set_xlabel("Adam step")
    axis.set_ylabel("loss")
    axis.set_title("Direct VPINN: Euclidean vs. Riesz loss")
    axis.legend()
    axis.grid(True, which="both", linestyle=":", linewidth=0.5)

    output_dir = create_timestamped_output_directory(
        "experiments", "01_vpinns_poisson_direct"
    )
    save_plot(figure, output_dir / "training.png")
    print(f"Saved plots to {output_dir}/")


if __name__ == "__main__":
    main()
