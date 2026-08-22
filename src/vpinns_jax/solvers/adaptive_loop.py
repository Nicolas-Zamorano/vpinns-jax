"""Closed-loop ``SOLVE -> ESTIMATE -> MARK -> REFINE`` adaptive cycle (math reference Section 4.4).

Implements the cycle for the headline MF-VPINN case: **direct** trial
(``models.boundary.ConstrainedAnsatz``) against the mesh-free **patch** test
space, Euclidean loss, with the **jump-term reduction** (Section 4.3)
applying at ESTIMATE (no facet quadrature needed). REFINE uses jax-fem's own
Rivara LEB engine (``jax_fem.mesh.refine_mesh_longest_edge_bisection``).

Unlike classical FE AMR, the trial function does not depend on the mesh at
all: the same network parameters are **warm-started** into the next
refinement level rather than reinitialized, which is exactly what
``run_adaptive_loop`` does (the model returned from level ``k`` is the model
trained further at level ``k + 1``).

Other (trial, test space, loss) combinations reuse the same
train/estimate/mark/refine building blocks directly rather than this
specific orchestrator; see ``solvers.trainer.train_adam``,
``estimators.aposteriori``, ``estimators.marking``.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from jax_fem.mesh import TriangleMesh, refine_mesh_longest_edge_bisection

from vpinns_jax.estimators.aposteriori import (
    direct_trial_patch_residual_indicators,
    global_patch_estimator,
    patch_indicators_to_cell_indicators,
    patch_loss_indicators,
)
from vpinns_jax.estimators.marking import cap_marked_fraction, mark_by_dorfler_bulk_criterion
from vpinns_jax.losses.euclidean import euclidean_loss
from vpinns_jax.losses.residual import direct_trial_patch_residual
from vpinns_jax.losses.riesz import patch_self_energy, riesz_loss_patch
from vpinns_jax.models.boundary import ConstrainedAnsatz
from vpinns_jax.pde.base import EllipticPDE
from vpinns_jax.solvers.trainer import TrainingHistory, train_adam
from vpinns_jax.test_spaces.patch_space import create_patch_test_space

__all__ = ["AdaptiveIterationResult", "run_adaptive_loop"]


@dataclasses.dataclass(frozen=True, slots=True, eq=False)
class AdaptiveIterationResult:
    """One adaptive-loop iteration's outcome.

    Attributes
    ----------
    mesh : TriangleMesh
        The mesh this iteration solved on (*before* REFINE).
    model : eqx.Module
        The direct-trial network, trained at this level (warm-started from
        the previous level).
    history : TrainingHistory
    global_estimator : float
        ``eta = sqrt(sum_i eta_{L,i}^2 + eta_{R,i}^2)``.
    number_of_patches : int
    number_of_marked_cells : int
    """

    mesh: TriangleMesh
    model: eqx.Module
    history: TrainingHistory
    global_estimator: float
    number_of_patches: int
    number_of_marked_cells: int


def run_adaptive_loop(
    initial_mesh: TriangleMesh,
    problem_factory: Callable[[TriangleMesh], EllipticPDE],
    model: eqx.Module,
    distance_fn: Callable[[jax.Array], jax.Array],
    *,
    num_levels: int,
    use_riesz_loss: bool = False,
    lift: Callable[[jax.Array], jax.Array] | None = None,
    adam_steps: int = 2000,
    adam_learning_rate: float = 1e-3,
    dorfler_theta: float = 0.5,
    max_marked_fraction: float = 1.0,
    log_every: int | None = None,
) -> list[AdaptiveIterationResult]:
    """Run ``num_levels`` ``SOLVE -> ESTIMATE -> MARK -> REFINE`` iterations.

    Parameters
    ----------
    initial_mesh : TriangleMesh
        The level-0 mesh.
    problem_factory : Callable[[TriangleMesh], EllipticPDE]
        Rebuilds the PDE problem (in particular its ``dirichlet_conditions``,
        which reference mesh-specific boundary tags) for each refined mesh.
    model : eqx.Module
        The initial direct-trial network (e.g.
        ``models.ansatz.MultiLayerPerceptron``), wrapped internally in
        ``models.boundary.ConstrainedAnsatz(model, distance_fn, lift)`` for
        strong boundary imposition at every level.
    distance_fn : Callable[[jax.Array], jax.Array]
        ``B(x)`` for the ``ConstrainedAnsatz`` (e.g.
        ``models.boundary.ExactDistanceFunction``).
    num_levels : int
        Number of refinement levels (the loop trains ``num_levels`` times,
        refining after every level but the last).
    use_riesz_loss : bool
        Euclidean loss (default) or the patch-collapse Riesz loss.
    lift : Callable[[jax.Array], jax.Array] | None
        ``gbar(x)``, ``gbar|_Gamma = g``, for a **non-homogeneous** Dirichlet
        problem (``models.boundary.ConstrainedAnsatz``'s ``lift``). Defaults
        to the zero function, correct only for a homogeneous problem (e.g.
        ``pde.create_poisson_sin_sin_problem``) -- pass the problem's own
        ``exact_solution`` (or any ``H^1`` extension of its boundary data)
        for a non-homogeneous one (e.g.
        ``pde.create_poisson_singular_problem``), or the loop will silently
        solve the wrong (homogeneous) problem.
    adam_steps, adam_learning_rate, log_every :
        Forwarded to ``solvers.trainer.train_adam`` at every level.
    dorfler_theta : float
        Doerfler bulk-marking fraction.
    max_marked_fraction : float
        Caps the marked-cell fraction each level (``estimators.marking
        .cap_marked_fraction``).

    Returns
    -------
    list[AdaptiveIterationResult]
        One entry per level, in order.
    """
    mesh = initial_mesh
    results: list[AdaptiveIterationResult] = []

    for level in range(num_levels):
        problem = problem_factory(mesh)
        patches = create_patch_test_space(mesh)

        def loss_fn(
            trial_model: eqx.Module,
            patches=patches,
            problem=problem,
            distance_fn=distance_fn,
            lift=lift,
        ) -> jax.Array:
            trial = ConstrainedAnsatz(trial_model, distance_fn, lift)
            residual = direct_trial_patch_residual(trial, patches, problem)
            if use_riesz_loss:
                self_energy = patch_self_energy(patches, problem)
                return riesz_loss_patch(residual, self_energy)
            return euclidean_loss(residual)

        model, history = train_adam(
            loss_fn,
            model,
            num_steps=adam_steps,
            learning_rate=adam_learning_rate,
            log_every=log_every,
        )

        trial = ConstrainedAnsatz(model, distance_fn, lift)
        residual = direct_trial_patch_residual(trial, patches, problem)
        self_energy = patch_self_energy(patches, problem)
        loss_indicators = patch_loss_indicators(residual, self_energy)
        residual_indicators = direct_trial_patch_residual_indicators(trial, patches, problem)
        eta = global_patch_estimator(loss_indicators, residual_indicators)

        patch_indicators = jnp.sqrt(loss_indicators**2 + residual_indicators**2)
        cell_indicators = patch_indicators_to_cell_indicators(patches, patch_indicators)
        marked = mark_by_dorfler_bulk_criterion(cell_indicators, dorfler_theta)
        marked = cap_marked_fraction(marked, cell_indicators, max_marked_fraction)

        results.append(
            AdaptiveIterationResult(
                mesh=mesh,
                model=model,
                history=history,
                global_estimator=float(eta),
                number_of_patches=patches.number_of_test_functions,
                number_of_marked_cells=int(jnp.sum(marked)),
            )
        )

        if level < num_levels - 1:
            mesh = refine_mesh_longest_edge_bisection(mesh, marked)

    return results
