"""
Strong imposition of Dirichlet conditions on neural trial functions.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jax_fem.function import FiniteElementFunction

from vpinns_jax.models.evaluation import Network, evaluate_network_values
from vpinns_jax.test_spaces.fe_space import FETestSpace

# Vertices of the L-shaped domain [-1, 1]^2 \ [0, 1] x [-1, 0], counterclockwise
# from the re-entrant corner, as in jax_fem.mesh.create_gmsh_l_shaped_mesh.
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
    """
    Distance from a point to the boundary of a polygon.

    Parameters
    ----------
    x : jax.Array
        Point, shape ``(2,)``.
    vertices : jax.Array
        Polygon vertices in order, shape ``(N_vertices, 2)``.

    Returns
    -------
    distance : jax.Array
        Unsigned distance to the nearest edge, shape ``()``.
    """
    edge_starts = vertices
    edges = jnp.roll(vertices, -1, axis=-2) - edge_starts
    parameters = jnp.sum((x - edge_starts) * edges, axis=-1) / jnp.sum(
        edges * edges, axis=-1
    )
    closest_points = (
        edge_starts + jnp.expand_dims(jnp.clip(parameters, 0.0, 1.0), axis=-1) * edges
    )
    return jnp.min(jnp.linalg.norm(x - closest_points, axis=-1), axis=-1)


class ExactDistanceFunction(eqx.Module):
    """
    Nonnegative function vanishing exactly on the boundary of a standard domain.

    Attributes
    ----------
    geometry : str
        ``"unit_square"``, ``"circle"`` (unit disk), or ``"l_shape"``
        (``[-1, 1]^2 \\ [0, 1] x [-1, 0]``).

    Notes
    -----
    The unit square uses ``x (1 - x) y (1 - y)`` and the unit disk
    ``1 - |x|^2``, both smooth. The L-shape uses the distance to the polygon
    boundary, which is only Lipschitz but is zero exactly on the boundary and
    positive inside, as strong imposition requires.
    """

    geometry: str = eqx.field(static=True)

    def __init__(self, geometry: str = "unit_square") -> None:
        """
        Initialize the distance function.

        Parameters
        ----------
        geometry : str
            One of ``"unit_square"``, ``"circle"``, ``"l_shape"``.
        """
        if geometry not in _GEOMETRIES:
            raise ValueError(
                f"geometry must be one of {_GEOMETRIES}, got {geometry!r}."
            )
        self.geometry = geometry

    def __call__(self, x: jax.Array) -> jax.Array:
        """
        Evaluate the distance function at one point.

        Parameters
        ----------
        x : jax.Array
            Point, shape ``(2,)``.

        Returns
        -------
        value : jax.Array
            Function value, shape ``()``.
        """
        if self.geometry == "unit_square":
            return x[0] * (1.0 - x[0]) * x[1] * (1.0 - x[1])
        if self.geometry == "circle":
            return 1.0 - jnp.sum(x * x, axis=-1)
        return _polygon_boundary_distance(x, jnp.asarray(_L_SHAPE_VERTICES))


def _zero_lift(x: jax.Array) -> jax.Array:
    """
    Zero lift for homogeneous Dirichlet data.

    Parameters
    ----------
    x : jax.Array
        Point, shape ``(d,)``.

    Returns
    -------
    value : jax.Array
        Zero, shape ``()``.
    """
    return jnp.zeros((), dtype=x.dtype)


class ConstrainedAnsatz(eqx.Module):
    """
    Network trial function satisfying a Dirichlet condition by construction.

    Attributes
    ----------
    network : Network
        Free network ``u_theta``.
    distance : Network
        Function ``B`` vanishing on the Dirichlet boundary.
    lift : Network
        Extension ``g_bar`` of the Dirichlet data into the domain.

    Notes
    -----
    The trial function is ``w = B u_theta + g_bar``. Since ``B = 0`` and
    ``g_bar = g`` on the Dirichlet boundary, ``w = g`` there for every
    ``u_theta``. The default lift is zero, which is correct only for
    homogeneous data; otherwise pass, e.g., the exact solution.
    """

    network: Network
    distance: Network
    lift: Network

    def __init__(
        self, network: Network, distance: Network, lift: Network | None = None
    ) -> None:
        """
        Initialize the ansatz.

        Parameters
        ----------
        network : Network
            Free network ``u_theta``.
        distance : Network
            Function ``B`` vanishing on the Dirichlet boundary.
        lift : Network | None
            Extension ``g_bar`` of the Dirichlet data; zero if ``None``.
        """
        self.network = network
        self.distance = distance
        self.lift = lift if lift is not None else _zero_lift

    def __call__(self, x: jax.Array) -> jax.Array:
        """
        Evaluate the trial function at one point.

        Parameters
        ----------
        x : jax.Array
            Point, shape ``(d,)``.

        Returns
        -------
        value : jax.Array
            ``B(x) u_theta(x) + g_bar(x)``, shape ``()``.
        """
        return self.distance(x) * self.network(x) + self.lift(x)


def interpolate_network(
    network: Network, fe_test_space: FETestSpace
) -> FiniteElementFunction:
    """
    Nodal interpolant of a network with the Dirichlet values imposed.

    Parameters
    ----------
    network : Network
        Scalar function of one point.
    fe_test_space : FETestSpace
        Test space whose nodal space and Dirichlet data define the interpolant.

    Returns
    -------
    interpolant : FiniteElementFunction
        ``sum_{i free} w(x_i) phi_i + sum_{i Dirichlet} g(x_i) phi_i``, in
        ``fe_test_space.basis.space``.

    Notes
    -----
    The network is evaluated at every DOF coordinate and the Dirichlet entries
    are overwritten, so the interpolant satisfies the Dirichlet condition at
    the DOFs exactly. The space must be nodal. The function only reads arrays,
    so it may be called inside a jitted loss.
    """
    space = fe_test_space.basis.space
    values = jnp.squeeze(
        evaluate_network_values(network, space.dof_coordinates), axis=-2
    )
    dof_values = values.at[..., fe_test_space.dirichlet_dofs, :].set(
        fe_test_space.dirichlet_values
    )
    return FiniteElementFunction(space=space, dof_values=dof_values)
