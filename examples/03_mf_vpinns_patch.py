"""Mesh-Free VPINN (MF-VPINN) on the Poisson sin(pi x) sin(pi y) problem.

Trial: direct, strong BC through the network ansatz.
Test space: mesh-free, vertex-star patches (``test_spaces.patch_space``),
*not* a global FE assembly -- every residual/loss/estimator quantity here is
a small local gather-and-reduce over each patch's own incident cells.
Loss: the robust, single-DOF-per-patch Riesz collapse
``sum_i r_h(w, varphi_i)^2 / (varphi_i, varphi_i)_A`` (math reference
Section 2.3's "single-function-per-patch identity"), compared against the
plain per-patch Euclidean loss.
"""

from __future__ import annotations

import time

import jax.random as jr
import matplotlib

matplotlib.use("Agg")

from jax_fem.mesh import create_gmsh_unit_square_mesh

from vpinns_jax.analysis import (
    direct_trial_energy_error,
    direct_trial_relative_l2_error,
)
from vpinns_jax.estimators import (
    direct_trial_patch_residual_indicators,
    global_patch_estimator,
    patch_loss_indicators,
)
from vpinns_jax.losses import (
    direct_trial_patch_residual,
    euclidean_loss,
    patch_self_energy,
    riesz_loss_patch,
)
from vpinns_jax.models import (
    ConstrainedAnsatz,
    ExactDistanceFunction,
    MultiLayerPerceptron,
)
from vpinns_jax.pde import create_poisson_sin_sin_problem
from vpinns_jax.solvers import train_adam
from vpinns_jax.test_spaces import create_fe_test_space, create_patch_test_space


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
    patches = create_patch_test_space(mesh)
    fe = create_fe_test_space(
        mesh, problem
    )  # only reused as a quadrature grid for exact errors
    distance = ExactDistanceFunction("unit_square")
    self_energy = patch_self_energy(patches, problem)

    print(
        f"MF-VPINN, Poisson sin-sin, {cells_per_side}x{cells_per_side} background mesh, "
        f"{patches.number_of_test_functions} patches (max valence {patches.patch_cells.shape[1]}), "
        f"cover constants (c_U2, c_U1) = ({patches.cover_constant_lower:.4f}, "
        f"{patches.cover_constant_upper:.4f})."
    )

    for loss_name in ("euclidean", "riesz"):
        network = MultiLayerPerceptron(width=width, depth=depth, key=jr.PRNGKey(seed))

        def loss_fn(trial_model, loss_name=loss_name):
            trial = ConstrainedAnsatz(trial_model, distance)
            residual = direct_trial_patch_residual(trial, patches, problem)
            if loss_name == "riesz":
                return riesz_loss_patch(residual, self_energy)
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

        residual = direct_trial_patch_residual(trial, patches, problem)
        loss_indicators = patch_loss_indicators(residual, self_energy)
        residual_indicators = direct_trial_patch_residual_indicators(
            trial, patches, problem
        )
        eta = float(global_patch_estimator(loss_indicators, residual_indicators))

        relative_l2 = float(direct_trial_relative_l2_error(trial, fe.basis, problem))
        energy = float(direct_trial_energy_error(trial, fe.basis, problem))
        print(
            f"[{loss_name:9s}] trained in {elapsed:.2f}s | final loss {float(history.loss[-1]):.4e} | "
            f"relative L2 {relative_l2:.4e} | energy {energy:.4e} | estimator eta {eta:.4e} | "
            f"efficiency eta/energy {eta / energy:.3f}"
        )


if __name__ == "__main__":
    main()
