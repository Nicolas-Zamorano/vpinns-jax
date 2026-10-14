"""
Patch test spaces integrated patch by patch, without a global mesh.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import sparse as jax_sparse
from jax_fem.assembly import GlobalMatrix, integrate_cellwise
from jax_fem.forms import elliptic_energy_bilinear_form
from jax_fem.problem import (
    EllipticCoefficientValues,
    EllipticProblem,
    evaluate_elliptic_coefficients,
)
from jax_fem.reference_cell import QuadratureRule, ReferenceTriangle

from vpinns_jax.test_spaces.patch_space import PatchFunctions

#: Raw weight of a patch at one point ``x: (d,)``, given the patch centre
#: ``(d,)`` and its integer kind ``()``; returns ``()``.
type PatchWeight = Callable[[jax.Array, jax.Array, jax.Array], jax.Array]


@dataclasses.dataclass(frozen=True, slots=True, eq=False)
class MeshfreePatchFamily:
    """
    Patches with closed-form raw weights, each with its own triangulation.

    Attributes
    ----------
    weight : PatchWeight
        Raw weight ``phi_i`` of a patch, from its centre and kind.
    centres : jax.Array
        Patch centres, shape ``(N_patch, d)``.
    kinds : jax.Array
        Integer kind of each patch, passed to ``weight``, shape
        ``(N_patch,)``.
    local_triangles : jax.Array
        Vertices of the triangles of each patch, padded by repeating its
        first triangle, shape ``(N_patch, T, 3, d)``.
    triangle_mask : jax.Array
        ``True`` for real triangles, shape ``(N_patch, T)``.
    neighbours : jax.Array
        Patches whose weight does not vanish identically on each patch, the
        patch itself first, padded with ``0``, shape ``(N_patch, C)``.
    neighbour_mask : jax.Array
        ``True`` for real neighbours, shape ``(N_patch, C)``.
    patch_diameters : jax.Array
        Diameter ``h_i`` of each patch, shape ``(N_patch,)``.
    boundary_patch_mask : jax.Array
        ``True`` for the boundary patches ``I_Gamma``, shape ``(N_patch,)``.
    interior_patches : jax.Array
        The interior patches ``I_Omega`` in increasing order, shape
        ``(N_interior,)``.

    Notes
    -----
    Each patch ``P_i`` is the union of its local triangles, and the weight of
    every neighbour is smooth on each of them, so its kinks lie on triangle
    edges. Integrals over ``P_i`` use these triangles alone: the family needs
    no global mesh. The neighbours give the covering function
    ``S = sum_j phi_j`` on ``P_i`` and the nonzero Gram matrix entries of row
    ``i``.
    """

    weight: PatchWeight
    centres: jax.Array
    kinds: jax.Array
    local_triangles: jax.Array
    triangle_mask: jax.Array
    neighbours: jax.Array
    neighbour_mask: jax.Array
    patch_diameters: jax.Array
    boundary_patch_mask: jax.Array
    interior_patches: jax.Array

    @property
    def number_of_patches(self) -> int:
        """
        Number of patches.

        Returns
        -------
        number_of_patches : int
            ``N_patch``, interior and boundary patches together.
        """
        return self.centres.shape[-2]


def create_meshfree_patch_family(
    weight: PatchWeight,
    centres: np.ndarray,
    kinds: np.ndarray,
    local_triangles: np.ndarray,
    triangle_mask: np.ndarray,
    neighbours: np.ndarray,
    neighbour_mask: np.ndarray,
    boundary_patch_mask: np.ndarray,
) -> MeshfreePatchFamily:
    """
    Validate a patch family and derive its padding, diameters, and index split.

    Parameters
    ----------
    weight : PatchWeight
        Raw weight of a patch, from its centre and kind.
    centres : np.ndarray
        Patch centres, shape ``(N_patch, d)``.
    kinds : np.ndarray
        Integer kind of each patch, shape ``(N_patch,)``.
    local_triangles : np.ndarray
        Triangles of each patch, shape ``(N_patch, T, 3, d)``; entries
        outside ``triangle_mask`` are ignored.
    triangle_mask : np.ndarray
        ``True`` for real triangles, shape ``(N_patch, T)``.
    neighbours : np.ndarray
        Neighbour patches, the patch itself first, shape ``(N_patch, C)``;
        entries outside ``neighbour_mask`` are ignored.
    neighbour_mask : np.ndarray
        ``True`` for real neighbours, shape ``(N_patch, C)``.
    boundary_patch_mask : np.ndarray
        ``True`` for the boundary patches, shape ``(N_patch,)``.

    Returns
    -------
    family : MeshfreePatchFamily
        The family, with padding triangles replaced by the first real
        triangle of their patch.

    Raises
    ------
    ValueError
        If a patch has no triangle or does not list itself as its first
        neighbour.
    """
    number_of_patches = centres.shape[-2]
    triangle_mask = np.asarray(triangle_mask, dtype=bool)
    neighbour_mask = np.asarray(neighbour_mask, dtype=bool)
    if not np.all(np.any(triangle_mask, axis=-1)):
        raise ValueError("Every patch needs at least one local triangle.")
    neighbours = np.where(neighbour_mask, neighbours, 0)
    if not (
        np.all(neighbour_mask[:, 0])
        and np.array_equal(neighbours[:, 0], np.arange(number_of_patches))
    ):
        raise ValueError("Every patch must list itself as its first neighbour.")

    first_triangles = np.take_along_axis(
        local_triangles,
        np.expand_dims(np.argmax(triangle_mask, axis=-1), axis=(-1, -2, -3)),
        axis=-3,
    )
    local_triangles = np.where(
        np.expand_dims(triangle_mask, axis=(-1, -2)), local_triangles, first_triangles
    )
    # Padding repeats real triangles, so it does not change the diameters.
    differences = np.expand_dims(local_triangles, axis=(-2, -3)) - np.expand_dims(
        local_triangles, axis=(-4, -5)
    )
    patch_diameters = np.sqrt(
        np.max(np.sum(differences**2, axis=-1), axis=(-1, -2, -3, -4))
    )
    boundary_patch_mask = np.asarray(boundary_patch_mask, dtype=bool)
    return MeshfreePatchFamily(
        weight=weight,
        centres=jnp.asarray(centres, dtype=float),
        kinds=jnp.asarray(kinds, dtype=jnp.int32),
        local_triangles=jnp.asarray(local_triangles, dtype=float),
        triangle_mask=jnp.asarray(triangle_mask),
        neighbours=jnp.asarray(neighbours, dtype=jnp.int32),
        neighbour_mask=jnp.asarray(neighbour_mask),
        patch_diameters=jnp.asarray(patch_diameters),
        boundary_patch_mask=jnp.asarray(boundary_patch_mask),
        interior_patches=jnp.flatnonzero(jnp.asarray(~boundary_patch_mask)),
    )


@dataclasses.dataclass(frozen=True, slots=True, eq=False)
class MeshfreePatchTestSpace:
    """
    Patch test space evaluated on the local quadrature of every patch.

    Attributes
    ----------
    family : MeshfreePatchFamily
        Patches, raw weights, and local triangulations.
    points : jax.Array
        Quadrature points of each patch, shape ``(N_patch, P, 1, d)``.
    weights : jax.Array
        Quadrature weights of each patch, zero on padding triangles, shape
        ``(N_patch, P, 1, 1)``.
    coefficients : EllipticCoefficientValues
        Coefficients at ``points``.
    test_functions : PatchFunctions
        Test function ``psi_i`` of each patch at its own points: values,
        shape ``(N_patch, P, 1, 1)``, and gradients, shape
        ``(N_patch, P, 1, d)``.
    raw_weights : PatchFunctions
        Raw weight ``phi_i`` of each patch at its own points, same shapes.
    gram : GlobalMatrix
        Energy Gram matrix ``(psi_i, psi_j)_A`` of all patches, shape
        ``(N_patch, N_patch)``.
    covering_bounds : tuple[float, float]
        Minimum and maximum of ``S = sum_j phi_j`` over the vertices and
        quadrature points of the local triangles.

    Notes
    -----
    Row ``i`` of ``points``, ``weights``, ``test_functions`` and
    ``raw_weights`` belongs to the patch ``P_i``, so the residual
    ``a(w, psi_i) - l(psi_i)`` is the integral of the patch's own row and
    needs no assembly: patches play the role of cells with one local
    function each. The network is evaluated on every patch separately, so
    points in the overlap of several patches are evaluated once per patch.
    """

    family: MeshfreePatchFamily
    points: jax.Array
    weights: jax.Array
    coefficients: EllipticCoefficientValues
    test_functions: PatchFunctions
    raw_weights: PatchFunctions
    gram: GlobalMatrix
    covering_bounds: tuple[float, float]


def _local_quadrature(
    family: MeshfreePatchFamily, quadrature: QuadratureRule, measure: float
) -> tuple[jax.Array, jax.Array]:
    """
    Quadrature points and weights on the local triangles of every patch.

    Parameters
    ----------
    family : MeshfreePatchFamily
        Planar patch family.
    quadrature : QuadratureRule
        Rule on the reference triangle.
    measure : float
        Area of the reference triangle.

    Returns
    -------
    points : jax.Array
        Points, shape ``(N_patch, T Q, 1, 2)``.
    weights : jax.Array
        Weights, zero on padding triangles, shape ``(N_patch, T Q, 1, 1)``.
    """
    triangles = np.asarray(family.local_triangles)
    number_of_patches, number_of_triangles = triangles.shape[:2]
    origins = triangles[..., :1, :]
    edges = triangles[..., 1:, :] - origins
    reference_points = np.squeeze(np.asarray(quadrature.points), axis=(0, 2))
    reference_weights = np.squeeze(np.asarray(quadrature.weights), axis=(0, 2, 3))
    number_of_points = number_of_triangles * reference_weights.shape[-1]
    points = origins + reference_points @ edges
    areas = measure * np.abs(np.linalg.det(edges))
    weights = reference_weights * np.expand_dims(
        np.where(np.asarray(family.triangle_mask), areas, 0.0), axis=-1
    )
    return (
        jnp.asarray(np.reshape(points, (number_of_patches, number_of_points, 1, 2))),
        jnp.asarray(np.reshape(weights, (number_of_patches, number_of_points, 1, 1))),
    )


def _evaluate_neighbour_weights(
    family: MeshfreePatchFamily, points: jax.Array
) -> PatchFunctions:
    """
    Raw weights of the neighbours of every patch at the patch's points.

    Parameters
    ----------
    family : MeshfreePatchFamily
        Patch family.
    points : jax.Array
        Points of each patch, shape ``(N_patch, P, 1, d)``.

    Returns
    -------
    neighbour_weights : PatchFunctions
        Values, shape ``(N_patch, P, C, 1)``, and gradients, shape
        ``(N_patch, P, C, d)``, zero at padding neighbours; the patch's own
        weight is in slot ``0``.
    """
    centres = jnp.take(family.centres, family.neighbours, axis=-2)
    kinds = jnp.take(family.kinds, family.neighbours, axis=-1)
    values, gradients = jnp.vectorize(
        jax.value_and_grad(family.weight), signature="(d),(d),()->(),(d)"
    )(points, jnp.expand_dims(centres, axis=-3), jnp.expand_dims(kinds, axis=-2))
    mask = jnp.expand_dims(family.neighbour_mask, axis=-2)
    return PatchFunctions(
        values=jnp.expand_dims(jnp.where(mask, values, 0.0), axis=-1),
        gradients=jnp.where(jnp.expand_dims(mask, axis=-1), gradients, 0.0),
    )


def _own_functions(functions: PatchFunctions) -> PatchFunctions:
    """
    The patch's own function from a neighbour layout.

    Parameters
    ----------
    functions : PatchFunctions
        Neighbour functions, shapes ``(N_patch, P, C, 1)`` and
        ``(N_patch, P, C, d)``.

    Returns
    -------
    own : PatchFunctions
        Slot ``0``, shapes ``(N_patch, P, 1, 1)`` and ``(N_patch, P, 1, d)``.
    """
    return PatchFunctions(
        values=functions.values[..., :1, :], gradients=functions.gradients[..., :1, :]
    )


def create_meshfree_patch_test_space(
    family: MeshfreePatchFamily,
    problem: EllipticProblem,
    *,
    quadrature_degree: int = 2,
    normalise: bool = False,
) -> MeshfreePatchTestSpace:
    """
    Evaluate the patch functions on their local quadrature and their Gram matrix.

    Parameters
    ----------
    family : MeshfreePatchFamily
        Planar patch family.
    problem : EllipticProblem
        Problem providing the coefficients; only its coefficient functions
        are used.
    quadrature_degree : int
        Exactness degree of the rule on every local triangle.
    normalise : bool
        Use the Shepard-normalised weights ``psi_i = phi_i / S`` as test
        functions; if ``False``, use the raw weights.

    Returns
    -------
    patch_test_space : MeshfreePatchTestSpace
        The test space.

    Notes
    -----
    The raw weights and their gradients come from ``family.weight`` by
    automatic differentiation, at points inside the local triangles where
    every weight is smooth. With normalisation, ``S`` on a patch is the sum of
    its neighbours' weights and ``grad psi_i = grad phi_i / S -
    phi_i grad S / S^2``. Row ``i`` of the Gram matrix is integrated over
    ``P_i`` against the neighbours of ``i``; two patches whose triangles agree
    on their overlap therefore get a symmetric pair of entries.
    """
    reference_cell = ReferenceTriangle()
    points, weights = _local_quadrature(
        family,
        reference_cell.create_quadrature(quadrature_degree),
        reference_cell.measure,
    )
    neighbour_weights = _evaluate_neighbour_weights(family, points)
    if normalise:
        sums = jnp.sum(neighbour_weights.values, axis=-2, keepdims=True)
        sum_gradients = jnp.sum(neighbour_weights.gradients, axis=-2, keepdims=True)
        neighbour_functions = PatchFunctions(
            values=neighbour_weights.values / sums,
            gradients=neighbour_weights.gradients / sums
            - neighbour_weights.values * sum_gradients / sums**2,
        )
    else:
        neighbour_functions = neighbour_weights
    test_functions = _own_functions(neighbour_functions)

    coefficients = evaluate_elliptic_coefficients(problem, points)
    local_gram = jnp.squeeze(
        integrate_cellwise(
            elliptic_energy_bilinear_form(
                test_functions, neighbour_functions, coefficients
            ),
            weights,
        ),
        axis=(-3, -2),
    )
    rows = jnp.broadcast_to(
        jnp.expand_dims(jnp.arange(family.number_of_patches), axis=-1),
        family.neighbours.shape,
    )
    gram = jax_sparse.BCOO(
        (
            jnp.ravel(jnp.where(family.neighbour_mask, local_gram, 0.0)),
            jnp.stack((jnp.ravel(rows), jnp.ravel(family.neighbours)), axis=-1),
        ),
        shape=(family.number_of_patches, family.number_of_patches),
    )

    triangles = family.local_triangles
    vertices = jnp.reshape(
        triangles,
        (family.number_of_patches, 3 * triangles.shape[-3], 1, triangles.shape[-1]),
    )
    sums = jnp.concatenate(
        (
            jnp.sum(neighbour_weights.values, axis=-2),
            jnp.sum(_evaluate_neighbour_weights(family, vertices).values, axis=-2),
        ),
        axis=-2,
    )
    return MeshfreePatchTestSpace(
        family=family,
        points=points,
        weights=weights,
        coefficients=coefficients,
        test_functions=test_functions,
        raw_weights=_own_functions(neighbour_weights),
        gram=gram,
        covering_bounds=(float(jnp.min(sums)), float(jnp.max(sums))),
    )
