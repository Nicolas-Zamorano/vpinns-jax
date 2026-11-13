"""
Gradient-based training of network trial functions.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from tqdm import tqdm

#: A function of the model evaluated during training, returning a pytree of
#: scalars, e.g. from ``vpinns_jax.analysis.create_training_monitor``.
type Monitor = Callable[[eqx.Module], Any]


@dataclasses.dataclass(frozen=True, slots=True, eq=False)
class TrainingHistory:
    """
    Quantities recorded during training.

    Attributes
    ----------
    loss : jax.Array
        Loss before each step, shape ``(number_of_steps,)``.
    monitored_steps : jax.Array
        Steps at which the monitor was evaluated, shape
        ``(number_of_records,)``; step ``n`` is the model after ``n`` updates.
    monitored : Any
        Monitor outputs stacked along a leading axis of length
        ``number_of_records``, or ``None`` if no monitor was given.
    """

    loss: jax.Array
    monitored_steps: jax.Array = dataclasses.field(
        default_factory=lambda: jnp.zeros((0,), dtype=int)
    )
    monitored: Any = None


# Monitor fields shown on the progress bar, with their labels.
_PROGRESS_FIELDS = (("relative_energy_error", "rel. error"), ("robustness", "ratio"))


def _progress_fields(record: Any) -> dict[str, str]:
    """
    Monitor values shown on the progress bar.

    Parameters
    ----------
    record : Any
        Monitor output.

    Returns
    -------
    fields : dict[str, str]
        The relative energy error and the ratio ``sqrt(L) / |u - w|_A`` of a
        record with the ``relative_energy_error`` and ``robustness`` fields,
        such as a ``TrainingRecord``, by label; the fields it lacks are left
        out.
    """
    return {
        label: f"{float(getattr(record, name)):.3e}"
        for name, label in _PROGRESS_FIELDS
        if hasattr(record, name)
    }


def _run_steps(
    step: Callable[[Any], tuple[Any, jax.Array]],
    carry: Any,
    *,
    num_steps: int,
    log_every: int | None,
    early_stopping_patience: int | None,
    early_stopping_min_delta: float,
    monitor: Callable[[Any], Any] | None,
    monitor_every: int,
    description: str,
) -> tuple[Any, TrainingHistory]:
    """
    Apply a training step repeatedly and record the loss and the monitor.

    Parameters
    ----------
    step : Callable[[Any], tuple[Any, jax.Array]]
        Jitted step returning the next carry and the loss at the current one.
    carry : Any
        Initial parameters and optimizer state.
    num_steps : int
        Maximum number of steps, at least one.
    log_every : int | None
        Show a progress bar whose loss is updated every this many steps and
        at the last step; ``0`` or ``None`` hides it.
    early_stopping_patience : int | None
        Stop after this many consecutive steps without improvement; ``None``
        disables early stopping.
    early_stopping_min_delta : float
        Decrease of the best loss that counts as an improvement.
    monitor : Callable[[Any], Any] | None
        Function of the carry evaluated every ``monitor_every`` steps and after
        the last step; ``None`` disables monitoring.
    monitor_every : int
        Number of steps between monitor evaluations, at least one.
    description : str
        Label of the progress bar.

    Returns
    -------
    carry : Any
        Final parameters and optimizer state.
    history : TrainingHistory
        Loss of every step taken and the monitor records.

    Notes
    -----
    The progress bar shows the loss and, if the monitor returns records with
    the ``relative_energy_error`` and ``robustness`` fields, such as those of
    ``vpinns_jax.analysis.create_training_monitor``, the relative energy error
    and the ratio ``sqrt(L) / |u - w|_A`` of the latest record.

    Early stopping and the loss on the progress bar are read on the host,
    which synchronizes at those steps; without them the steps are dispatched
    asynchronously.
    """
    if num_steps < 1:
        raise ValueError(f"num_steps must be at least 1, got {num_steps}.")
    if monitor_every < 1:
        raise ValueError(f"monitor_every must be at least 1, got {monitor_every}.")
    if log_every is not None and log_every < 0:
        raise ValueError(f"log_every must be non-negative, got {log_every}.")
    show_progress = bool(log_every)
    losses = []
    records = []
    recorded_steps = []
    monitored = {}
    best_loss = math.inf
    steps_without_improvement = 0
    with tqdm(
        total=num_steps,
        desc=description,
        unit="step",
        dynamic_ncols=True,
        disable=not show_progress,
    ) as progress:
        for index in range(num_steps):
            if monitor is not None and index % monitor_every == 0:
                records.append(monitor(carry))
                recorded_steps.append(index)
                if show_progress:
                    monitored = _progress_fields(records[-1])
            carry, loss = step(carry)
            losses.append(loss)
            progress.update()
            if show_progress and (index % log_every == 0 or index == num_steps - 1):
                progress.set_postfix(
                    {"loss": f"{float(loss):.3e}", **monitored}, refresh=False
                )
            if early_stopping_patience is not None:
                if float(loss) < best_loss - early_stopping_min_delta:
                    best_loss = float(loss)
                    steps_without_improvement = 0
                else:
                    steps_without_improvement += 1
                    if steps_without_improvement >= early_stopping_patience:
                        break
        if monitor is None:
            return carry, TrainingHistory(loss=jnp.stack(losses))
        records.append(monitor(carry))
        recorded_steps.append(len(losses))
        if show_progress:
            progress.set_postfix(
                {"loss": f"{float(losses[-1]):.3e}", **_progress_fields(records[-1])}
            )
    return carry, TrainingHistory(
        loss=jnp.stack(losses),
        monitored_steps=jnp.asarray(recorded_steps),
        monitored=jax.tree.map(lambda *values: jnp.stack(values), *records),
    )


def _train(
    loss_fn: Callable[[eqx.Module], jax.Array],
    model: eqx.Module,
    optimizer: optax.GradientTransformationExtraArgs,
    create_step: Callable[
        [Callable[[eqx.Module], jax.Array]],
        Callable[[Any], tuple[Any, jax.Array]],
    ],
    *,
    monitor: Monitor | None,
    **run_options: Any,
) -> tuple[eqx.Module, TrainingHistory]:
    """
    Partition a model, run an optimizer on its parameters, and rebuild it.

    Parameters
    ----------
    loss_fn : Callable[[eqx.Module], jax.Array]
        Scalar loss of the model.
    model : eqx.Module
        Model whose floating-point arrays are the parameters.
    optimizer : optax.GradientTransformationExtraArgs
        Optimizer whose state is initialized from the parameters.
    create_step : Callable
        Builds the jitted step from the loss as a function of the parameters.
    monitor : Monitor | None
        Function of the model to record, or ``None``.
    **run_options : Any
        Remaining options of ``_run_steps``.

    Returns
    -------
    model : eqx.Module
        Trained model.
    history : TrainingHistory
        Loss of every step taken and the monitor records.
    """
    parameters, static = eqx.partition(model, eqx.is_inexact_array)

    def objective(parameters: eqx.Module) -> jax.Array:
        """
        Loss as a function of the parameters.

        Parameters
        ----------
        parameters : eqx.Module
            Floating-point arrays of the model.

        Returns
        -------
        loss : jax.Array
            Loss, shape ``()``.
        """
        return loss_fn(eqx.combine(parameters, static))

    @eqx.filter_jit
    def carry_monitor(carry: tuple[eqx.Module, Any]) -> Any:
        """
        Monitor of the model held in the carry.

        Parameters
        ----------
        carry : tuple[eqx.Module, Any]
            Parameters and optimizer state.

        Returns
        -------
        record : Any
            Monitor output for the current model.
        """
        return monitor(eqx.combine(carry[0], static))

    (parameters, _), history = _run_steps(
        create_step(objective),
        (parameters, optimizer.init(parameters)),
        monitor=carry_monitor if monitor is not None else None,
        **run_options,
    )
    return eqx.combine(parameters, static), history


def _adam_learning_rate(
    learning_rate: float, final_learning_rate: float | None, num_steps: int
) -> float | optax.Schedule:
    """
    Constant or exponentially decaying Adam learning rate.

    Parameters
    ----------
    learning_rate : float
        Learning rate, or its initial value.
    final_learning_rate : float | None
        Learning rate after ``num_steps`` steps; ``None`` keeps it constant.
    num_steps : int
        Number of steps of the decay.

    Returns
    -------
    learning_rate : float | optax.Schedule
        ``learning_rate``, or the schedule
        ``k -> learning_rate (final_learning_rate / learning_rate)^(k / num_steps)``.

    Raises
    ------
    ValueError
        If a learning rate is not positive.
    """
    if learning_rate <= 0.0:
        raise ValueError(f"learning_rate must be positive, got {learning_rate}.")
    if final_learning_rate is None:
        return learning_rate
    if final_learning_rate <= 0.0:
        raise ValueError(
            f"final_learning_rate must be positive, got {final_learning_rate}."
        )
    return optax.exponential_decay(
        init_value=learning_rate,
        transition_steps=max(num_steps, 1),
        decay_rate=final_learning_rate / learning_rate,
    )


def train_adam(
    loss_fn: Callable[[eqx.Module], jax.Array],
    model: eqx.Module,
    *,
    num_steps: int,
    learning_rate: float = 1e-3,
    final_learning_rate: float | None = None,
    log_every: int | None = None,
    early_stopping_patience: int | None = None,
    early_stopping_min_delta: float = 0.0,
    monitor: Monitor | None = None,
    monitor_every: int = 100,
) -> tuple[eqx.Module, TrainingHistory]:
    """
    Train a model with Adam.

    Parameters
    ----------
    loss_fn : Callable[[eqx.Module], jax.Array]
        Scalar loss as a function of the model alone; other data is closed
        over.
    model : eqx.Module
        Model whose floating-point arrays are the parameters.
    num_steps : int
        Maximum number of steps.
    learning_rate : float
        Adam learning rate, or its initial value with ``final_learning_rate``.
    final_learning_rate : float | None
        Learning rate reached after ``num_steps`` steps, from
        ``learning_rate`` by exponential decay; ``None`` keeps the learning
        rate constant.
    log_every : int | None
        Show a progress bar whose loss is updated every this many steps,
        with the relative energy error and the ratio ``sqrt(L) / |u - w|_A``
        of the latest record of a ``create_training_monitor`` monitor; ``0``
        or ``None`` hides it.
    early_stopping_patience : int | None
        Stop after this many consecutive steps without improvement; ``None``
        disables early stopping.
    early_stopping_min_delta : float
        Decrease of the best loss that counts as an improvement.
    monitor : Monitor | None
        Function of the model recorded every ``monitor_every`` steps and after
        the last step; ``None`` disables monitoring.
    monitor_every : int
        Number of steps between monitor evaluations.

    Returns
    -------
    model : eqx.Module
        Trained model.
    history : TrainingHistory
        Loss of every step taken and the monitor records.

    Notes
    -----
    With ``final_learning_rate``, step ``k`` uses the learning rate
    ``learning_rate (final_learning_rate / learning_rate)^(k / num_steps)``.
    """
    optimizer = optax.adam(
        _adam_learning_rate(learning_rate, final_learning_rate, num_steps)
    )

    def create_step(
        objective: Callable[[eqx.Module], jax.Array],
    ) -> Callable[[Any], tuple[Any, jax.Array]]:
        """
        Build the jitted Adam step.

        Parameters
        ----------
        objective : Callable[[eqx.Module], jax.Array]
            Loss as a function of the parameters.

        Returns
        -------
        step : Callable[[Any], tuple[Any, jax.Array]]
            Step mapping parameters and optimizer state to their update and
            the loss before it.
        """

        @eqx.filter_jit
        def step(
            carry: tuple[eqx.Module, Any],
        ) -> tuple[tuple[eqx.Module, Any], jax.Array]:
            """
            One Adam step.

            Parameters
            ----------
            carry : tuple[eqx.Module, Any]
                Parameters and optimizer state.

            Returns
            -------
            carry : tuple[eqx.Module, Any]
                Updated parameters and optimizer state.
            loss : jax.Array
                Loss before the update, shape ``()``.
            """
            parameters, state = carry
            loss, gradients = jax.value_and_grad(objective)(parameters)
            updates, state = optimizer.update(gradients, state, parameters)
            return (optax.apply_updates(parameters, updates), state), loss

        return step

    return _train(
        loss_fn,
        model,
        optimizer,
        create_step,
        num_steps=num_steps,
        log_every=log_every,
        early_stopping_patience=early_stopping_patience,
        early_stopping_min_delta=early_stopping_min_delta,
        monitor=monitor,
        monitor_every=monitor_every,
        description="Adam",
    )


def train_lbfgs(
    loss_fn: Callable[[eqx.Module], jax.Array],
    model: eqx.Module,
    *,
    num_steps: int,
    memory_size: int = 10,
    log_every: int | None = None,
    early_stopping_patience: int | None = None,
    early_stopping_min_delta: float = 0.0,
    monitor: Monitor | None = None,
    monitor_every: int = 100,
) -> tuple[eqx.Module, TrainingHistory]:
    """
    Train a model with L-BFGS.

    Parameters
    ----------
    loss_fn : Callable[[eqx.Module], jax.Array]
        Scalar loss as a function of the model alone; other data is closed
        over.
    model : eqx.Module
        Model whose floating-point arrays are the parameters.
    num_steps : int
        Maximum number of steps.
    memory_size : int
        Number of past updates kept for the curvature approximation.
    log_every : int | None
        Show a progress bar whose loss is updated every this many steps,
        with the relative energy error and the ratio ``sqrt(L) / |u - w|_A``
        of the latest record of a ``create_training_monitor`` monitor; ``0``
        or ``None`` hides it.
    early_stopping_patience : int | None
        Stop after this many consecutive steps without improvement; ``None``
        disables early stopping.
    early_stopping_min_delta : float
        Decrease of the best loss that counts as an improvement.
    monitor : Monitor | None
        Function of the model recorded every ``monitor_every`` steps and after
        the last step; ``None`` disables monitoring.
    monitor_every : int
        Number of steps between monitor evaluations.

    Returns
    -------
    model : eqx.Module
        Trained model.
    history : TrainingHistory
        Loss of every step taken and the monitor records.

    Notes
    -----
    Uses ``optax.lbfgs`` with its default zoom line search, which evaluates
    the loss several times per step; ``optax.value_and_grad_from_state``
    reuses the line search's last evaluation.
    """
    optimizer = optax.lbfgs(memory_size=memory_size)

    def create_step(
        objective: Callable[[eqx.Module], jax.Array],
    ) -> Callable[[Any], tuple[Any, jax.Array]]:
        """
        Build the jitted L-BFGS step.

        Parameters
        ----------
        objective : Callable[[eqx.Module], jax.Array]
            Loss as a function of the parameters.

        Returns
        -------
        step : Callable[[Any], tuple[Any, jax.Array]]
            Step mapping parameters and optimizer state to their update and
            the loss before it.
        """
        value_and_grad = optax.value_and_grad_from_state(objective)

        @eqx.filter_jit
        def step(
            carry: tuple[eqx.Module, Any],
        ) -> tuple[tuple[eqx.Module, Any], jax.Array]:
            """
            One L-BFGS step.

            Parameters
            ----------
            carry : tuple[eqx.Module, Any]
                Parameters and optimizer state.

            Returns
            -------
            carry : tuple[eqx.Module, Any]
                Updated parameters and optimizer state.
            loss : jax.Array
                Loss before the update, shape ``()``.
            """
            parameters, state = carry
            loss, gradients = value_and_grad(parameters, state=state)
            updates, state = optimizer.update(
                gradients,
                state,
                parameters,
                value=loss,
                grad=gradients,
                value_fn=objective,
            )
            return (optax.apply_updates(parameters, updates), state), loss

        return step

    return _train(
        loss_fn,
        model,
        optimizer,
        create_step,
        num_steps=num_steps,
        log_every=log_every,
        early_stopping_patience=early_stopping_patience,
        early_stopping_min_delta=early_stopping_min_delta,
        monitor=monitor,
        monitor_every=monitor_every,
        description="L-BFGS",
    )


def _concatenate_histories(
    first: TrainingHistory, second: TrainingHistory
) -> TrainingHistory:
    """
    Join the histories of two consecutive training stages.

    Parameters
    ----------
    first : TrainingHistory
        History of the first stage.
    second : TrainingHistory
        History of the stage that continued from the first stage's model.

    Returns
    -------
    history : TrainingHistory
        Losses and records of both stages, with the second stage's steps
        offset by the first stage's number of steps. The second stage's
        initial record duplicates the first stage's final one and is dropped.
    """
    loss = jnp.concatenate((first.loss, second.loss), axis=-1)
    if first.monitored is None or second.monitored is None:
        return TrainingHistory(loss=loss)
    offset = first.loss.shape[-1]
    return TrainingHistory(
        loss=loss,
        monitored_steps=jnp.concatenate(
            (first.monitored_steps, second.monitored_steps[1:] + offset), axis=-1
        ),
        monitored=jax.tree.map(
            lambda head, tail: jnp.concatenate((head, tail[1:]), axis=0),
            first.monitored,
            second.monitored,
        ),
    )


def train(
    loss_fn: Callable[[eqx.Module], jax.Array],
    model: eqx.Module,
    *,
    adam_steps: int,
    adam_learning_rate: float = 1e-3,
    adam_final_learning_rate: float | None = None,
    lbfgs_steps: int = 0,
    log_every: int | None = None,
    monitor: Monitor | None = None,
    monitor_every: int = 100,
) -> tuple[eqx.Module, TrainingHistory]:
    """
    Train a model with Adam, optionally followed by L-BFGS.

    Parameters
    ----------
    loss_fn : Callable[[eqx.Module], jax.Array]
        Scalar loss as a function of the model alone.
    model : eqx.Module
        Model whose floating-point arrays are the parameters.
    adam_steps : int
        Number of Adam steps.
    adam_learning_rate : float
        Adam learning rate, or its initial value with
        ``adam_final_learning_rate``.
    adam_final_learning_rate : float | None
        Adam learning rate after ``adam_steps`` steps, reached by exponential
        decay; ``None`` keeps it constant.
    lbfgs_steps : int
        Number of L-BFGS steps after Adam; none if ``0``.
    log_every : int | None
        Show a progress bar whose loss is updated every this many steps,
        with the relative energy error and the ratio ``sqrt(L) / |u - w|_A``
        of the latest record of a ``create_training_monitor`` monitor; ``0``
        or ``None`` hides it.
    monitor : Monitor | None
        Function of the model recorded during both stages; ``None`` disables
        monitoring.
    monitor_every : int
        Number of steps between monitor evaluations.

    Returns
    -------
    model : eqx.Module
        Trained model.
    history : TrainingHistory
        Losses and records of the Adam steps followed by those of the
        L-BFGS steps.
    """
    model, history = train_adam(
        loss_fn,
        model,
        num_steps=adam_steps,
        learning_rate=adam_learning_rate,
        final_learning_rate=adam_final_learning_rate,
        log_every=log_every,
        monitor=monitor,
        monitor_every=monitor_every,
    )
    if lbfgs_steps > 0:
        model, lbfgs_history = train_lbfgs(
            loss_fn,
            model,
            num_steps=lbfgs_steps,
            log_every=log_every,
            monitor=monitor,
            monitor_every=monitor_every,
        )
        history = _concatenate_histories(history, lbfgs_history)
    return model, history
