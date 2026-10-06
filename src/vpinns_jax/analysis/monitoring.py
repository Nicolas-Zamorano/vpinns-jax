"""
Monitoring of losses and errors during training.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp


class TrainingRecord(NamedTuple):
    """
    Loss and error quantities of a model at one training step.

    Attributes
    ----------
    loss : jax.Array
        Loss ``L``, the sum of the terms, shape ``(...,)``.
    terms : Mapping[str, jax.Array]
        Named terms of the loss, each of shape ``(...,)``.
    energy_error : jax.Array
        ``|u - w|_A``, shape ``(...,)``.
    relative_loss : jax.Array
        ``sqrt(L) / |u|_A``, shape ``(...,)``.
    relative_energy_error : jax.Array
        ``|u - w|_A / |u|_A``, shape ``(...,)``.
    robustness : jax.Array
        ``sqrt(L) / |u - w|_A``, the ratio of the two relative quantities,
        shape ``(...,)``.

    Notes
    -----
    A monitor returns scalars; in a ``TrainingHistory`` every field gains a
    leading axis over the recorded steps. The losses are squared norms, so
    ``sqrt(L)`` is compared with the energy error. A loss that is a robust
    error estimator keeps ``robustness`` bounded above and below during the
    whole training, not only at its end.
    """

    loss: jax.Array
    terms: Mapping[str, jax.Array]
    energy_error: jax.Array
    relative_loss: jax.Array
    relative_energy_error: jax.Array
    robustness: jax.Array


def create_training_monitor(
    loss_terms: Callable[[eqx.Module], Mapping[str, jax.Array]],
    energy_error: Callable[[eqx.Module], jax.Array],
    solution_energy_norm: jax.Array,
) -> Callable[[eqx.Module], TrainingRecord]:
    """
    Monitor recording the loss, its terms, and the energy error of a model.

    Parameters
    ----------
    loss_terms : Callable[[eqx.Module], Mapping[str, jax.Array]]
        Named terms of the loss as functions of the model; the loss is their
        sum.
    energy_error : Callable[[eqx.Module], jax.Array]
        Energy error ``|u - w|_A`` of the trial function built from the model,
        e.g. through ``compute_network_error`` or ``compute_energy_error``.
    solution_energy_norm : jax.Array
        ``|u|_A``, e.g. from ``compute_solution_norm``.

    Returns
    -------
    monitor : Callable[[eqx.Module], TrainingRecord]
        Function of the model to pass as ``monitor`` to the trainers.

    Notes
    -----
    Defining the loss as the sum of ``loss_terms`` keeps the recorded terms
    consistent with what is optimized. The monitor evaluates the loss again
    and the error on the error quadrature, so it is called only every few
    steps.
    """

    def monitor(model: eqx.Module) -> TrainingRecord:
        """
        Loss and error quantities of the model.

        Parameters
        ----------
        model : eqx.Module
            The model being trained.

        Returns
        -------
        record : TrainingRecord
            Its loss, loss terms, and errors.
        """
        terms = dict(loss_terms(model))
        loss = sum(terms.values(), start=jnp.zeros(()))
        error = energy_error(model)
        root_loss = jnp.sqrt(loss)
        return TrainingRecord(
            loss=loss,
            terms=terms,
            energy_error=error,
            relative_loss=root_loss / solution_energy_norm,
            relative_energy_error=error / solution_energy_norm,
            robustness=root_loss / error,
        )

    return monitor
