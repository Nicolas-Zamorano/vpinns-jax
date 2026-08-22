"""Boundary condition imposition for neural trial functions (math reference Section 3).

Two strong-imposition schemes are provided:

- ``ConstrainedAnsatz``: strong imposition *through the network ansatz*,
  ``u_theta^g = B * u_theta + gbar``, ``B|_Gamma = 0``, ``gbar|_Gamma = g``.
  Used with the **direct** trial representation ``w_theta = u_theta^g``.
- ``precompute_interpolation_data`` / ``interpolated_trial_dofs``: strong
  imposition *on the interpolant*, ``w_theta = Pi_h u_theta = sum_{i in
  interior} u_theta(x_i) phi_i + sum_{i in Gamma} g(x_i) phi_i``. Used with
  the **interpolated** trial representation ``w_theta = Pi_h u_theta``.

``precompute_interpolation_data`` is split from ``interpolated_trial_dofs``
because ``FiniteElementSpace``/``EllipticProblem`` are plain dataclasses, not
registered JAX pytrees: they cannot be passed into a ``jax.jit``/
``eqx.filter_jit``-wrapped function directly. ``precompute_interpolation_data``
does the one-time, eager DOF classification; ``interpolated_trial_dofs`` is
then a pure function of plain arrays, safe to call from inside a jitted
training step.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jax_fem.constraints import evaluate_dirichlet_dof_values
from jax_fem.function import FiniteElementFunction
from jax_fem.space import FiniteElementSpace

from vpinns_jax.pde.base import EllipticPDE

__all__ = [
    "ExactDistanceFunction",
    "ConstrainedAnsatz",
    "InterpolationData",
    "precompute_interpolation_data",
    "interpolated_trial_dofs",
    "interpolated_trial_function",
]

# Vertices of the re-entrant L-shaped domain [-1,1]^2 \ [0,1]x[-1,0], CCW
# from the re-entrant corner -- matches jax_fem.mesh.gmsh's own L-shape.
_L_SHAPE_VERTICES = (
    (0.0, 0.0),
    (1.0, 0.0),
    (1.0, 1.0),
    (-1.0, 1.0),
    (-1.0, -1.0),
    (0.0, -1.0),
)

_GEOMETRIES = ("unit_square", "circle", "l_shape")


def _polygon_boundary_distance(x: jax.Array, vertices: jax.Array) -> jax.Array:
    """Unsigned distance from ``x`` to the boundary of the polygon ``vertices`` (CCW)."""
    edge_start = vertices
    edge_end = jnp.roll(vertices, -1, axis=0)
    edge_vector = edge_end - edge_start
    edge_length_squared = jnp.sum(edge_vector * edge_vector, axis=-1)
    projection = jnp.sum((x[None, :] - edge_start) * edge_vector, axis=-1) / (
        edge_length_squared
    )
    projection = jnp.clip(projection, 0.0, 1.0)
    closest_point = edge_start + projection[:, None] * edge_vector
    distances = jnp.linalg.norm(x[None, :] - closest_point, axis=-1)
    return jnp.min(distances)


class ExactDistanceFunction(eqx.Module):
    """Smooth field ``B(x) >= 0``, ``B|_Gamma = 0``, for standard geometries.

    - ``"unit_square"``: ``B(x, y) = x(1-x) y(1-y)`` -- the classical smooth
      product distance field for ``[0, 1]^2``.
    - ``"circle"``: ``B(x) = 1 - |x|^2`` for the unit disk.
    - ``"l_shape"``: unsigned point-to-polygon-boundary distance for
      ``[-1,1]^2 \\ [0,1]x[-1,0]`` -- not a smooth closed form (the boundary
      has corners, including the re-entrant one), but exactly zero on
      ``Gamma`` and strictly positive in the interior, which is all strong
      imposition requires.
    """

    geometry: str = eqx.field(static=True)

    def __init__(self, geometry: str = "unit_square"):
        if geometry not in _GEOMETRIES:
            raise ValueError(f"geometry must be one of {_GEOMETRIES}, got {geometry!r}.")
        self.geometry = geometry

    def __call__(self, x: jax.Array) -> jax.Array:
        if self.geometry == "unit_square":
            return x[0] * (1.0 - x[0]) * x[1] * (1.0 - x[1])
        if self.geometry == "circle":
            return 1.0 - jnp.sum(x * x)
        return _polygon_boundary_distance(x, jnp.asarray(_L_SHAPE_VERTICES))


class ConstrainedAnsatz(eqx.Module):
    r"""Strong Dirichlet imposition through the network ansatz.

    ``u_theta^g(x) = B(x) u_theta(x) + gbar(x)``, ``B|_Gamma = 0``,
    ``gbar|_Gamma = g`` (math reference Section 3): ``u_theta^g|_Gamma = g``
    exactly, by construction, for *any* ``u_theta``. Used as the **direct**
    trial representation ``w_theta = u_theta^g``.

    ``lift`` defaults to the zero function (homogeneous Dirichlet data).
    """

    network: Callable[[jax.Array], jax.Array]
    distance: Callable[[jax.Array], jax.Array]
    lift: Callable[[jax.Array], jax.Array]

    def __init__(
        self,
        network: Callable[[jax.Array], jax.Array],
        distance: Callable[[jax.Array], jax.Array],
        lift: Callable[[jax.Array], jax.Array] | None = None,
    ):
        self.network = network
        self.distance = distance
        self.lift = lift if lift is not None else (lambda x: jnp.zeros(()))

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.distance(x) * self.network(x) + self.lift(x)


class InterpolationData(NamedTuple):
    """Plain-array DOF classification for ``interpolated_trial_dofs``.

    Precomputed once, eagerly, from a ``FiniteElementSpace``/``EllipticPDE``
    pair; every field is a JAX array or Python ``int``, safe to pass into a
    jitted function.
    """

    dof_coordinates: jax.Array  # (N_dofs, 2)
    free_dofs: jax.Array  # (N_free,)
    dirichlet_dofs: jax.Array  # (N_dirichlet,)
    dirichlet_values: jax.Array  # (N_dirichlet, 1)
    number_of_dofs: int


def precompute_interpolation_data(
    space: FiniteElementSpace, problem: EllipticPDE
) -> InterpolationData:
    """Classify ``space``'s DOFs into free (interior) vs. Dirichlet (boundary).

    Free DOFs are exactly the interior test space's DOF set ``I_int`` (test
    functions in ``V_h subset H^1_0``); Dirichlet DOFs get ``problem``'s
    boundary data evaluated at their coordinate.
    """
    dirichlet_dofs, dirichlet_values = evaluate_dirichlet_dof_values(problem, space)
    is_dirichlet = (
        jnp.zeros((space.number_of_dofs,), dtype=bool).at[dirichlet_dofs].set(True)
    )
    free_dofs = jnp.nonzero(~is_dirichlet)[0]
    dof_coordinates = space.dof_coordinates[:, 0, :]
    return InterpolationData(
        dof_coordinates=dof_coordinates,
        free_dofs=free_dofs,
        dirichlet_dofs=dirichlet_dofs,
        dirichlet_values=dirichlet_values,
        number_of_dofs=space.number_of_dofs,
    )


def interpolated_trial_dofs(
    network: Callable[[jax.Array], jax.Array], data: InterpolationData
) -> jax.Array:
    r"""Coordinate interpolation of ``w_theta = Pi_h u_theta`` with a strong BC.

    .. math::

        \Pi_h u_\theta = \sum_{i \in I_{int}} u_\theta(x_i) \varphi_i
            + \sum_{i \in I_\Gamma} g(x_i) \varphi_i

    Interior DOFs get the network's value at their coordinate; boundary DOFs
    get the exact Dirichlet value at their coordinate -- so the boundary
    condition holds exactly, with no boundary loss term. Pure function of
    plain arrays: safe to call inside a jitted training step.

    Returns
    -------
    jax.Array
        The interpolant's global DOF vector, shape ``(N_dofs, 1)``.
    """
    interior_values = jax.vmap(network)(data.dof_coordinates[data.free_dofs])
    dof_values = jnp.zeros((data.number_of_dofs,))
    dof_values = dof_values.at[data.free_dofs].set(interior_values)
    dof_values = dof_values.at[data.dirichlet_dofs].set(data.dirichlet_values[:, 0])
    return dof_values[:, None]


def interpolated_trial_function(
    network: Callable[[jax.Array], jax.Array],
    space: FiniteElementSpace,
    data: InterpolationData,
) -> FiniteElementFunction:
    """Convenience: ``interpolated_trial_dofs`` wrapped as a ``FiniteElementFunction``.

    Not jit-safe (constructs a ``FiniteElementFunction``, which embeds
    ``space``, a non-pytree dataclass); use for diagnostics/plotting after
    training, not inside a training step.
    """
    return FiniteElementFunction(
        space=space, dof_values=interpolated_trial_dofs(network, data)
    )
