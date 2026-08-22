"""``vpinns-jax``: Mesh-Free and Robust Variational Neural Networks (MF-RVPINNs) on ``jax-fem``.

See ``context/implementation_plan.md`` for the full architecture and
``context/MF-RVPINNs_math_reference.md`` for the underlying mathematics.
Submodules (``pde``, ``models``, ``test_spaces``, ``losses``, ``estimators``,
``solvers``, ``analysis``) can be imported directly for their complete APIs;
this top-level module re-exports the core symbols named in the
implementation plan's Component 8.
"""

from vpinns_jax.estimators import (
    cap_marked_fraction,
    global_patch_estimator,
    mark_by_dorfler_bulk_criterion,
)
from vpinns_jax.estimators import (
    interpolated_trial_patch_residual_indicators as aposteriori_estimator,
)
from vpinns_jax.models import ConstrainedAnsatz, InterpolationData, interpolated_trial_dofs
from vpinns_jax.pde import EllipticPDE
from vpinns_jax.pde import create_poisson_sin_sin_problem as PoissonProblem
from vpinns_jax.solvers import AdaptiveIterationResult, TrainingHistory, train, train_adam
from vpinns_jax.solvers import run_adaptive_loop as AdaptiveLoop
from vpinns_jax.test_spaces import PatchTestSpace, create_patch_test_space

from vpinns_jax.losses import (  # isort: skip
    euclidean_loss as EuclideanLoss,
)
from vpinns_jax.losses import (  # isort: skip
    riesz_loss_global,
    riesz_loss_patch,
)


def RieszLoss(residual, *, gram_free_dense=None, self_energy=None):
    """``L_Riesz(w_theta) = r^T G^{-1} r`` -- dispatches on which of the two
    keyword-only arguments is given: ``gram_free_dense`` for the global
    (FE test space) Gram-matrix inversion (``losses.riesz.riesz_loss_global``),
    ``self_energy`` for the single-DOF patch collapse
    (``losses.riesz.riesz_loss_patch``). Exactly one must be given.
    """
    if (gram_free_dense is None) == (self_energy is None):
        raise ValueError("Pass exactly one of gram_free_dense or self_energy.")
    if gram_free_dense is not None:
        return riesz_loss_global(residual, gram_free_dense)
    return riesz_loss_patch(residual, self_energy)


__all__ = [
    "EllipticPDE",
    "PoissonProblem",
    "ConstrainedAnsatz",
    "InterpolationData",
    "interpolated_trial_dofs",
    "PatchTestSpace",
    "create_patch_test_space",
    "EuclideanLoss",
    "RieszLoss",
    "aposteriori_estimator",
    "global_patch_estimator",
    "mark_by_dorfler_bulk_criterion",
    "cap_marked_fraction",
    "TrainingHistory",
    "train",
    "train_adam",
    "AdaptiveIterationResult",
    "AdaptiveLoop",
]
