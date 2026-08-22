"""Multi-stage optimization solver: Adam, optionally followed by L-BFGS.

Deliberately decoupled from PDE/test-space specifics: ``train_adam``/
``train_lbfgs``/``train`` take a bare ``loss_fn: Callable[[eqx.Module],
jax.Array]``, so any auxiliary data (test space, problem, coefficients) is
captured by the caller's closure rather than threaded through this module.
This keeps the trainer reusable verbatim across every (trial, test space,
loss) combination in the library.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import optax

__all__ = ["TrainingHistory", "train_adam", "train_lbfgs", "train"]


@dataclasses.dataclass(frozen=True, slots=True, eq=False)
class TrainingHistory:
    """Per-step training loss.

    Attributes
    ----------
    loss : jax.Array
        Shape ``(num_steps,)``.
    """

    loss: jax.Array


def train_adam(
    loss_fn: Callable[[eqx.Module], jax.Array],
    model: eqx.Module,
    *,
    num_steps: int,
    learning_rate: float = 1e-3,
    log_every: int | None = None,
    early_stopping_patience: int | None = None,
    early_stopping_min_delta: float = 0.0,
) -> tuple[eqx.Module, TrainingHistory]:
    """Train ``model`` by minimizing ``loss_fn`` with Adam.

    Parameters
    ----------
    loss_fn : Callable[[eqx.Module], jax.Array]
        Scalar loss, a pure function of ``model`` alone (close over any
        other data).
    model : eqx.Module
        The (trainable) model to optimize; its array leaves are the
        parameters.
    num_steps : int
        Number of Adam steps.
    learning_rate : float
    log_every : int | None
        Print the loss every this many steps, plus the first and last. ``None``
        disables logging.
    early_stopping_patience : int | None
        Stop once ``num_steps`` consecutive steps pass without the loss
        improving by at least ``early_stopping_min_delta``. ``None``
        disables early stopping (and avoids a per-step host sync).

    Returns
    -------
    model : eqx.Module
        The trained model.
    history : TrainingHistory
    """
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_jit
    def step(model: eqx.Module, opt_state):
        loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
        updates, opt_state = optimizer.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss

    loss_history = []
    best_loss = jnp.inf
    patience_counter = 0
    for i in range(num_steps):
        model, opt_state, loss = step(model, opt_state)
        loss_history.append(loss)
        if log_every is not None and (i % log_every == 0 or i == num_steps - 1):
            print(f"  step {i:6d}  loss {float(loss):.6e}")
        if early_stopping_patience is not None:
            loss_value = float(loss)
            if loss_value < best_loss - early_stopping_min_delta:
                best_loss = loss_value
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    break

    return model, TrainingHistory(loss=jnp.stack(loss_history))


def train_lbfgs(
    loss_fn: Callable[[eqx.Module], jax.Array],
    model: eqx.Module,
    *,
    max_iterations: int = 500,
    tolerance: float = 1e-8,
):
    """Refine ``model`` with L-BFGS (``jaxopt``), starting from its current parameters.

    Requires the optional ``jaxopt`` dependency (not currently declared in
    ``pyproject.toml``).

    Returns
    -------
    model : eqx.Module
        The refined model.
    state : jaxopt L-BFGS optimizer state
        Final solver state (contains e.g. ``state.iter_num``, ``state.error``).
    """
    try:
        import jaxopt
    except ImportError as error:
        raise ImportError(
            "train_lbfgs requires the optional 'jaxopt' dependency (not "
            "currently declared in pyproject.toml). Install it with: "
            "pip install jaxopt."
        ) from error

    params, static = eqx.partition(model, eqx.is_array)

    def objective(params: eqx.Module) -> jax.Array:
        return loss_fn(eqx.combine(params, static))

    solver = jaxopt.LBFGS(fun=objective, maxiter=max_iterations, tol=tolerance)
    result = solver.run(params)
    return eqx.combine(result.params, static), result.state


def train(
    loss_fn: Callable[[eqx.Module], jax.Array],
    model: eqx.Module,
    *,
    adam_steps: int,
    adam_learning_rate: float = 1e-3,
    lbfgs_steps: int = 0,
    log_every: int | None = None,
) -> tuple[eqx.Module, TrainingHistory]:
    """Adam stage, optionally followed by an L-BFGS refinement stage (``lbfgs_steps > 0``)."""
    model, history = train_adam(
        loss_fn, model, num_steps=adam_steps, learning_rate=adam_learning_rate, log_every=log_every
    )
    if lbfgs_steps > 0:
        model, _ = train_lbfgs(loss_fn, model, max_iterations=lbfgs_steps)
    return model, history
