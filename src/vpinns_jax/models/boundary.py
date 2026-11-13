"""
Strong imposition of Dirichlet conditions on neural trial functions.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jax_fem.function import FiniteElementFunction
from jax_fem.problem import ScalarCoefficient

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

# Vertices of the unit square, counterclockwise from the origin.
_UNIT_SQUARE_VERTICES = ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))


def _boundary_data_values(
    boundary_value: ScalarCoefficient, points: jax.Array
) -> jax.Array:
    """
    Dirichlet data at points given without the row-vector axis.

    Parameters
    ----------
    boundary_value : ScalarCoefficient
        Dirichlet data ``g``; takes points of shape ``(*batch, 1, 2)`` and
        returns ``(*batch, 1, 1)``.
    points : jax.Array
        Boundary points, shape ``(..., 2)``.

    Returns
    -------
    values : jax.Array
        ``g`` at the points, shape ``(...)``.
    """
    values = boundary_value(jnp.expand_dims(points, axis=-2))
    return jnp.squeeze(values, axis=(-2, -1))


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


class TransfiniteLift(eqx.Module):
    """
    Extension of Dirichlet data into the unit square by transfinite interpolation.

    Attributes
    ----------
    boundary_value : ScalarCoefficient
        Dirichlet data ``g``; takes points of shape ``(*batch, 1, 2)`` and
        returns ``(*batch, 1, 1)``, as ``DirichletCondition.value`` does.

    Notes
    -----
    The lift is the Coons patch of ``g``: the linear interpolants of ``g``
    between the sides ``x = 0, 1`` and between the sides ``y = 0, 1``, minus
    the bilinear interpolant of its corner values,

    ``g_bar(x, y) = (1 - x) g(0, y) + x g(1, y) + (1 - y) g(x, 0)
    + y g(x, 1) - (1 - x) (1 - y) g(0, 0) - x (1 - y) g(1, 0)
    - (1 - x) y g(0, 1) - x y g(1, 1)``.

    It equals ``g`` on the whole boundary and evaluates ``g`` only there, so
    it carries no information about the solution inside the domain. It is as
    smooth as the restrictions of ``g`` to the sides, so the strong residual
    can differentiate it twice when they are twice differentiable. It
    reproduces exactly every sum of a function affine in ``x`` and one affine
    in ``y``, such as ``a(x) + y b(x) + x c(y)``.
    """

    boundary_value: ScalarCoefficient

    def __call__(self, x: jax.Array) -> jax.Array:
        """
        Evaluate the lift at one point.

        Parameters
        ----------
        x : jax.Array
            Point of the unit square, shape ``(2,)``.

        Returns
        -------
        value : jax.Array
            ``g_bar(x)``, shape ``()``.
        """
        ends = jnp.array([0.0, 1.0], dtype=x.dtype)
        # Weights of the ends 0 and 1 in the linear interpolation along each axis.
        x_weights = jnp.stack((1.0 - x[0], x[0]))
        y_weights = jnp.stack((1.0 - x[1], x[1]))
        on_vertical_sides = _boundary_data_values(
            self.boundary_value,
            jnp.stack((ends, jnp.broadcast_to(x[1], (2,))), axis=-1),
        )
        on_horizontal_sides = _boundary_data_values(
            self.boundary_value,
            jnp.stack((jnp.broadcast_to(x[0], (2,)), ends), axis=-1),
        )
        # at_corners[i, j] = g(ends[i], ends[j]).
        at_corners = _boundary_data_values(
            self.boundary_value,
            jnp.stack(jnp.meshgrid(ends, ends, indexing="ij"), axis=-1),
        )
        return (
            x_weights @ on_vertical_sides
            + y_weights @ on_horizontal_sides
            - x_weights @ at_corners @ y_weights
        )


class WachspressLift(eqx.Module):
    """
    Extension of Dirichlet data into the unit square by transfinite
    interpolation with Wachspress coordinates.

    Attributes
    ----------
    boundary_value : ScalarCoefficient
        Dirichlet data ``g``; takes points of shape ``(*batch, 1, 2)`` and
        returns ``(*batch, 1, 1)``, as ``DirichletCondition.value`` does.

    Notes
    -----
    Let ``v_0, ..., v_3`` be the vertices of the square in counterclockwise
    order from the origin, with Wachspress coordinates ``lambda_i``, which on
    the unit square are the bilinear shape functions. The lift is

    ``g_bar = sum_i lambda_i (g(a_i) + g(b_i) - g(v_i))``,

    where ``a_i = v_i + lambda_(i+1) (v_(i+1) - v_i)`` and
    ``b_i = v_i + lambda_(i-1) (v_(i-1) - v_i)`` lie on the two sides that
    meet at ``v_i``, indices modulo four. On the side from ``v_i`` to
    ``v_(i+1)`` only ``lambda_i`` and ``lambda_(i+1)`` are nonzero and they
    sum to one, while ``a_i = b_(i+1) = x``, ``b_i = v_i`` and
    ``a_(i+1) = v_(i+1)``, so ``g_bar = g`` on the whole boundary. The lift
    evaluates ``g`` only on the boundary and is as smooth as the
    restrictions of ``g`` to the sides. This is the transfinite interpolant
    of Computational Mechanics (2026) 78:1015-1044.

    When ``g`` is singular at a vertex, ``a_i`` and ``b_i`` approach that
    vertex along both sides, so the singularity spreads along the sides
    meeting there, as for ``TransfiniteLift``: data ``g ~ t^(2/3)`` at the
    origin give ``Lap g_bar ~ x^(-4/3)`` along ``x = 0``.
    """

    boundary_value: ScalarCoefficient

    def __call__(self, x: jax.Array) -> jax.Array:
        """
        Evaluate the lift at one point.

        Parameters
        ----------
        x : jax.Array
            Point of the unit square, shape ``(2,)``.

        Returns
        -------
        value : jax.Array
            ``g_bar(x)``, shape ``()``.
        """
        vertices = jnp.asarray(_UNIT_SQUARE_VERTICES, dtype=x.dtype)
        coordinates = jnp.stack(
            (
                (1.0 - x[0]) * (1.0 - x[1]),
                x[0] * (1.0 - x[1]),
                x[0] * x[1],
                (1.0 - x[0]) * x[1],
            )
        )
        following = vertices + jnp.expand_dims(jnp.roll(coordinates, -1), axis=-1) * (
            jnp.roll(vertices, -1, axis=-2) - vertices
        )
        preceding = vertices + jnp.expand_dims(jnp.roll(coordinates, 1), axis=-1) * (
            jnp.roll(vertices, 1, axis=-2) - vertices
        )
        # The vertex values are constant, so they are evaluated apart: no
        # derivative of g is taken at a vertex, where it may be singular.
        on_sides = _boundary_data_values(
            self.boundary_value, jnp.stack((following, preceding))
        )
        at_vertices = _boundary_data_values(self.boundary_value, vertices)
        return coordinates @ (on_sides[0] + on_sides[1] - at_vertices)


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
    homogeneous data; otherwise pass an extension of the data, e.g. a
    ``TransfiniteLift`` on the unit square.
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
