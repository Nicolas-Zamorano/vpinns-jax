"""Automated convergence study: uniform vs. adaptive refinement (implementation plan Section 7).

Runs the direct-trial + patch-test-space + Euclidean-loss MF-VPINN pipeline
(the same building blocks as ``solvers.adaptive_loop``) across several
refinement levels, either uniformly (every cell refined each level) or
adaptively (Doerfler-marked cells only), and records DOFs, exact errors
(``analysis.errors``), the global estimator, and the efficiency index at
each level.

Unlike ``solvers.adaptive_loop.run_adaptive_loop``, each level here trains a
**fresh** network (via ``model_factory``) rather than warm-starting: a
convergence study wants each level's reported error to reflect that level's
own approximation power, not residual optimization history carried over
from a coarser mesh.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
from jax_fem.mesh import TriangleMesh, refine_mesh_longest_edge_bisection

from vpinns_jax.analysis.errors import (
    direct_trial_energy_error,
    direct_trial_h1_seminorm_error,
    direct_trial_relative_l2_error,
    efficiency_index,
)
from vpinns_jax.estimators.aposteriori import (
    direct_trial_patch_residual_indicators,
    global_patch_estimator,
    patch_indicators_to_cell_indicators,
    patch_loss_indicators,
)
from vpinns_jax.estimators.marking import mark_by_dorfler_bulk_criterion
from vpinns_jax.losses.euclidean import euclidean_loss
from vpinns_jax.losses.residual import direct_trial_patch_residual
from vpinns_jax.losses.riesz import patch_self_energy
from vpinns_jax.models.boundary import ConstrainedAnsatz
from vpinns_jax.pde.base import EllipticPDE
from vpinns_jax.solvers.trainer import train_adam
from vpinns_jax.test_spaces.fe_space import create_fe_test_space
from vpinns_jax.test_spaces.patch_space import create_patch_test_space

if TYPE_CHECKING:
    from matplotlib.figure import Figure

__all__ = ["ConvergenceRecord", "run_convergence_study", "plot_convergence_comparison"]


@dataclasses.dataclass(frozen=True, slots=True, eq=False)
class ConvergenceRecord:
    """One refinement level's DOFs, exact errors, estimator, and efficiency.

    Attributes
    ----------
    level : int
    number_of_dofs : int
        ``N_dof`` (P1 vertex count) at this level.
    relative_l2_error, h1_seminorm_error, energy_error : float
        ``analysis.errors``' direct-trial error evaluators.
    global_estimator : float
        ``eta`` (``estimators.aposteriori.global_patch_estimator``).
    efficiency_index : float
        ``eta / energy_error``.
    """

    level: int
    number_of_dofs: int
    relative_l2_error: float
    h1_seminorm_error: float
    energy_error: float
    global_estimator: float
    efficiency_index: float


def run_convergence_study(
    initial_mesh: TriangleMesh,
    problem_factory: Callable[[TriangleMesh], EllipticPDE],
    model_factory: Callable[[jax.Array], eqx.Module],
    distance_fn: Callable[[jax.Array], jax.Array],
    *,
    num_levels: int,
    adaptive: bool,
    key: jax.Array,
    lift: Callable[[jax.Array], jax.Array] | None = None,
    adam_steps: int = 2000,
    adam_learning_rate: float = 1e-3,
    dorfler_theta: float = 0.5,
) -> list[ConvergenceRecord]:
    """Run ``num_levels`` refinement levels, uniform (``adaptive=False``) or Doerfler-adaptive.

    Parameters
    ----------
    initial_mesh : TriangleMesh
    problem_factory : Callable[[TriangleMesh], EllipticPDE]
        Rebuilds the PDE problem (its mesh-tag-dependent
        ``dirichlet_conditions``) for each refined mesh.
    model_factory : Callable[[jax.Array], eqx.Module]
        Builds a fresh direct-trial network from a PRNG key, once per level.
    distance_fn : Callable[[jax.Array], jax.Array]
        ``B(x)`` for the ``ConstrainedAnsatz`` strong-BC wrapper.
    num_levels : int
    adaptive : bool
        ``True``: Doerfler-mark by the patch estimator each level. ``False``:
        refine every cell each level (uniform refinement).
    key : jax.Array
        PRNG key; level ``i``'s network uses ``jax.random.fold_in(key, i)``.
    lift : Callable[[jax.Array], jax.Array] | None
        ``gbar(x)`` for a **non-homogeneous** Dirichlet problem
        (``models.boundary.ConstrainedAnsatz``'s ``lift``); defaults to zero,
        correct only for a homogeneous problem. See
        ``solvers.adaptive_loop.run_adaptive_loop``'s ``lift`` docstring.
    adam_steps, adam_learning_rate :
        Forwarded to ``solvers.trainer.train_adam`` at every level.
    dorfler_theta : float
        Doerfler bulk fraction, used only when ``adaptive=True``.

    Returns
    -------
    list[ConvergenceRecord]
        One entry per level, in order.
    """
    mesh = initial_mesh
    records: list[ConvergenceRecord] = []

    for level in range(num_levels):
        problem = problem_factory(mesh)
        patches = create_patch_test_space(mesh)
        fe_space = create_fe_test_space(mesh, problem)  # quadrature grid for exact-error evaluation

        model = model_factory(jax.random.fold_in(key, level))

        def loss_fn(
            trial_model: eqx.Module,
            patches=patches,
            problem=problem,
            distance_fn=distance_fn,
            lift=lift,
        ) -> jax.Array:
            trial = ConstrainedAnsatz(trial_model, distance_fn, lift)
            residual = direct_trial_patch_residual(trial, patches, problem)
            return euclidean_loss(residual)

        model, _ = train_adam(
            loss_fn, model, num_steps=adam_steps, learning_rate=adam_learning_rate
        )
        trial = ConstrainedAnsatz(model, distance_fn, lift)

        relative_l2 = float(direct_trial_relative_l2_error(trial, fe_space.basis, problem))
        h1_seminorm = float(direct_trial_h1_seminorm_error(trial, fe_space.basis, problem))
        energy = float(direct_trial_energy_error(trial, fe_space.basis, problem))

        residual = direct_trial_patch_residual(trial, patches, problem)
        self_energy = patch_self_energy(patches, problem)
        loss_indicators = patch_loss_indicators(residual, self_energy)
        residual_indicators = direct_trial_patch_residual_indicators(trial, patches, problem)
        eta = float(global_patch_estimator(loss_indicators, residual_indicators))

        records.append(
            ConvergenceRecord(
                level=level,
                number_of_dofs=fe_space.space.number_of_dofs,
                relative_l2_error=relative_l2,
                h1_seminorm_error=h1_seminorm,
                energy_error=energy,
                global_estimator=eta,
                efficiency_index=float(efficiency_index(eta, energy)),
            )
        )

        if level < num_levels - 1:
            if adaptive:
                patch_indicators = jnp.sqrt(loss_indicators**2 + residual_indicators**2)
                cell_indicators = patch_indicators_to_cell_indicators(patches, patch_indicators)
                marked = mark_by_dorfler_bulk_criterion(cell_indicators, dorfler_theta)
            else:
                marked = jnp.ones((mesh.cells_to_vertices.shape[0],), dtype=bool)
            mesh = refine_mesh_longest_edge_bisection(mesh, marked)

    return records


def plot_convergence_comparison(
    uniform_records: list[ConvergenceRecord],
    adaptive_records: list[ConvergenceRecord],
    *,
    error_attribute: str = "energy_error",
) -> Figure:
    """Log-log DOFs-vs-error comparison of uniform vs. adaptive refinement.

    Parameters
    ----------
    uniform_records, adaptive_records : list[ConvergenceRecord]
        From ``run_convergence_study(..., adaptive=False/True)``.
    error_attribute : str
        Which ``ConvergenceRecord`` field to plot (``"energy_error"``,
        ``"relative_l2_error"``, or ``"h1_seminorm_error"``).

    Returns
    -------
    figure : Figure
    """
    import matplotlib.pyplot as plt

    def _series(records: list[ConvergenceRecord]) -> tuple[jax.Array, jax.Array]:
        dofs = jnp.asarray([r.number_of_dofs for r in records], dtype=float)
        errors = jnp.asarray([getattr(r, error_attribute) for r in records], dtype=float)
        return dofs, errors

    uniform_dofs, uniform_errors = _series(uniform_records)
    adaptive_dofs, adaptive_errors = _series(adaptive_records)

    figure, axis = plt.subplots(figsize=(7, 5))
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("$N_{dofs}$")
    axis.set_ylabel(error_attribute.replace("_", " "))
    axis.set_title("Uniform vs. adaptive refinement")
    axis.grid(True, which="both", linestyle=":", linewidth=0.5)
    axis.plot(uniform_dofs, uniform_errors, "o-", color="tab:blue", label="uniform")
    axis.plot(adaptive_dofs, adaptive_errors, "s-", color="tab:orange", label="adaptive")
    axis.legend()
    return figure
