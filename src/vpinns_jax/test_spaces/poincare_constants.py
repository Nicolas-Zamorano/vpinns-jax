"""
Constants of the Poincare-Friedrichs inequalities (W2) and (W3) on single
patches, computed as eigenvalue problems with P1 finite elements.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Callable, Sequence
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse
import scipy.sparse.linalg
from jax_fem.assembly import GlobalMatrix, assemble_matrix, integrate_cellwise
from jax_fem.element import LagrangeTriangleP1
from jax_fem.reference_cell import ReferenceTriangle
from jax_fem.space import CellBasis, create_cell_basis, create_finite_element_space

from vpinns_jax.test_spaces.crisscross_mesh import create_crisscross_mesh

#: Weight of a reference patch at points of shape ``(..., 1, 2)``, returning
#: shape ``(..., 1, 1)``.
type ReferenceWeight = Callable[[jax.Array], jax.Array]

# Side of a rectangle by name: the fixed coordinate, and whether it is at the
# upper corner.
_RECTANGLE_SIDES = {
    "bottom": (1, False),
    "right": (0, True),
    "top": (1, True),
    "left": (0, False),
}

#: Reference configurations of the ``l^inf`` family at ``s = 1``: the lower
#: and upper corners of the patch, centred at the origin, and its Dirichlet
#: sides.
LINF_CONFIGURATIONS: dict[
    str, tuple[tuple[float, float], tuple[float, float], tuple[str, ...]]
] = {
    "interior": ((-1.0, -1.0), (1.0, 1.0), ()),
    "side": ((-1.0, 0.0), (1.0, 1.0), ("bottom",)),
    "corner": ((0.0, 0.0), (1.0, 1.0), ("bottom", "left")),
}


class PoincareConstants(NamedTuple):
    """
    Constants of (W2) and (W3) of a patch, as multiples of its scale.

    Attributes
    ----------
    weighted : float
        ``c_2(P) h_P`` of the weighted inequality (W2).
    unweighted : float
        ``C_P(P) h_P`` of the inequality (W3).
    diameter : float
        ``h_P``.

    Notes
    -----
    All three scale like the patch, so on a reference configuration of
    scale ``s = 1`` they are the multiples of ``s`` for every patch of that
    configuration.
    """

    weighted: float
    unweighted: float
    diameter: float


def to_scipy_matrix(matrix: GlobalMatrix) -> scipy.sparse.csr_array:
    """
    Convert an assembled ``jax_fem`` matrix to a SciPy sparse matrix.

    Parameters
    ----------
    matrix : GlobalMatrix
        Sparse matrix with repeated entries, shape ``(M, N)``.

    Returns
    -------
    matrix : scipy.sparse.csr_array
        The same matrix with the repeated entries summed, shape ``(M, N)``.
    """
    indices = np.asarray(matrix.indices)
    return scipy.sparse.csr_array(
        (np.asarray(matrix.data), (indices[:, 0], indices[:, 1])),
        shape=matrix.shape,
    )


def assemble_weighted_p1_matrices(
    basis: CellBasis, weight: jax.Array
) -> tuple[scipy.sparse.csr_array, scipy.sparse.csr_array]:
    """
    Weighted stiffness and mass matrices of a P1 space.

    Parameters
    ----------
    basis : CellBasis
        P1 basis at the quadrature points.
    weight : jax.Array
        Weight ``w`` at ``basis.physical_points``, shape ``(K, Q, 1, 1)``.

    Returns
    -------
    stiffness : scipy.sparse.csr_array
        ``(w grad phi_j, grad phi_i)``, shape ``(N, N)``.
    mass : scipy.sparse.csr_array
        ``(w phi_j, phi_i)``, shape ``(N, N)``.
    """
    space = basis.space
    stiffness = integrate_cellwise(
        weight * (basis.gradients @ basis.gradients.mT), basis.physical_weights
    )
    mass = integrate_cellwise(
        weight * (basis.values @ basis.values.mT), basis.physical_weights
    )
    return (
        to_scipy_matrix(assemble_matrix(stiffness, space, space)),
        to_scipy_matrix(assemble_matrix(mass, space, space)),
    )


def smallest_constrained_eigenvalue(
    stiffness: scipy.sparse.sparray,
    mass: scipy.sparse.sparray,
    *,
    fixed: np.ndarray | None = None,
    mean: np.ndarray | None = None,
) -> float:
    """
    Smallest eigenvalue of ``K x = lambda M x`` under a Dirichlet or a mean
    constraint.

    Parameters
    ----------
    stiffness : scipy.sparse.sparray
        Symmetric positive semidefinite ``K``, shape ``(N, N)``.
    mass : scipy.sparse.sparray
        Symmetric positive definite ``M``, shape ``(N, N)``.
    fixed : np.ndarray | None
        ``True`` for the entries fixed to zero, shape ``(N,)``.
    mean : np.ndarray | None
        Coefficients ``m`` of the constraint ``m^T x = 0``, shape ``(N,)``.

    Returns
    -------
    eigenvalue : float
        ``min x^T K x / x^T M x`` over the constrained ``x``.

    Raises
    ------
    ValueError
        Unless exactly one of ``fixed`` and ``mean`` is given.

    Notes
    -----
    The eigenvalue is found by shift-invert Lanczos with shift zero. With
    ``fixed``, the fixed rows and columns are removed and ``K`` must be
    definite on the rest. With ``mean``, ``K`` may vanish on the constants
    as long as ``m^T 1 != 0``: the inverse is the solve of the bordered
    system ``[[K, m], [m^T, 0]]``, which keeps every iterate in
    ``m^T x = 0``.
    """
    if (fixed is None) == (mean is None):
        raise ValueError("Pass exactly one of fixed and mean.")
    if fixed is not None:
        free = np.flatnonzero(~np.asarray(fixed, dtype=bool))
        stiffness = stiffness[free][:, free]
        mass = mass[free][:, free]
        factor = scipy.sparse.linalg.splu(scipy.sparse.csc_array(stiffness))

        def solve(right_hand_side: np.ndarray) -> np.ndarray:
            return factor.solve(np.ravel(right_hand_side))

    else:
        column = scipy.sparse.csc_array(np.expand_dims(mean, axis=-1))
        bordered = scipy.sparse.block_array(
            [[stiffness, column], [column.T, None]], format="csc"
        )
        factor = scipy.sparse.linalg.splu(bordered)

        def solve(right_hand_side: np.ndarray) -> np.ndarray:
            return factor.solve(np.append(np.ravel(right_hand_side), 0.0))[:-1]

    size = stiffness.shape[0]
    eigenvalues = scipy.sparse.linalg.eigsh(
        stiffness,
        k=1,
        M=mass,
        sigma=0.0,
        which="LM",
        OPinv=scipy.sparse.linalg.LinearOperator(
            (size, size), matvec=solve, dtype=float
        ),
        v0=np.random.default_rng(0).standard_normal(size),
        return_eigenvectors=False,
    )
    return float(eigenvalues[0])


def compute_rectangle_poincare_constants(
    lower: tuple[float, float],
    upper: tuple[float, float],
    weight: ReferenceWeight,
    dirichlet_sides: Sequence[str],
    refinement: int,
) -> PoincareConstants:
    """
    Constants of (W2) and (W3) of a rectangular patch.

    Parameters
    ----------
    lower : tuple[float, float]
        Lower-left corner of the patch.
    upper : tuple[float, float]
        Upper-right corner of the patch.
    weight : ReferenceWeight
        Weight ``phi`` of the patch, affine on every triangle of the
        criss-cross mesh with ``refinement`` cells per unit length.
    dirichlet_sides : Sequence[str]
        Sides among ``"bottom"``, ``"right"``, ``"top"`` and ``"left"`` on
        which ``v`` vanishes; empty for a patch of interior type.
    refinement : int
        Number of mesh cells per unit length; the sides must be integer
        multiples of ``1 / refinement``.

    Returns
    -------
    constants : PoincareConstants
        ``c_2 h``, ``C_P h`` and ``h`` of the patch, in its own units.

    Notes
    -----
    The constants are the inverse square roots of the smallest eigenvalues
    of ``(phi grad u, grad v) = lambda (phi u, v)`` for (W2) and of
    ``(grad u, grad v) = lambda (u, v)`` for (W3). For an interior patch, the
    functions have zero ``phi``-weighted mean, the constant ``v_i`` of the
    paper, in both problems; for a boundary patch, they vanish on the
    Dirichlet sides instead. P1 functions on the mesh form a subspace, so
    the discrete eigenvalues lie above the exact ones and the computed
    constants below them; they increase under refinement, since the
    criss-cross meshes are nested. The weight is affine on each triangle, so
    the degree-4 rule integrates the matrices exactly.
    """
    unknown = set(dirichlet_sides) - set(_RECTANGLE_SIDES)
    if unknown:
        raise ValueError(f"Unknown sides {sorted(unknown)}.")
    lower_corner = np.asarray(lower, dtype=float)
    upper_corner = np.asarray(upper, dtype=float)
    cells = np.rint((upper_corner - lower_corner) * refinement).astype(int)
    if not np.allclose(cells, (upper_corner - lower_corner) * refinement):
        raise ValueError(
            f"The sides of the patch {lower}--{upper} must be multiples of "
            f"1 / {refinement}."
        )
    mesh = create_crisscross_mesh(lower, upper, (int(cells[0]), int(cells[1])))
    element = LagrangeTriangleP1(reference_cell=ReferenceTriangle())
    basis = create_cell_basis(
        create_finite_element_space(mesh, element),
        element.reference_cell.create_quadrature(4),
    )
    weight_values = weight(basis.physical_points)
    weighted_stiffness, weighted_mass = assemble_weighted_p1_matrices(
        basis, weight_values
    )
    stiffness, mass = assemble_weighted_p1_matrices(basis, jnp.ones_like(weight_values))

    if dirichlet_sides:
        coordinates = np.reshape(np.asarray(basis.space.dof_coordinates), (-1, 2))
        fixed = np.zeros(coordinates.shape[0], dtype=bool)
        for side in dirichlet_sides:
            axis, at_upper = _RECTANGLE_SIDES[side]
            value = (upper_corner if at_upper else lower_corner)[axis]
            fixed |= np.isclose(coordinates[:, axis], value)
        constraint = {"fixed": fixed}
    else:
        weighted_masses = weighted_mass @ np.ones(weighted_mass.shape[0])
        constraint = {"mean": weighted_masses / np.sum(weighted_masses)}

    weighted = smallest_constrained_eigenvalue(
        weighted_stiffness, weighted_mass, **constraint
    )
    unweighted = smallest_constrained_eigenvalue(stiffness, mass, **constraint)
    return PoincareConstants(
        weighted=1.0 / math.sqrt(weighted),
        unweighted=1.0 / math.sqrt(unweighted),
        diameter=float(np.linalg.norm(upper_corner - lower_corner)),
    )


def _linf_reference_weight(points: jax.Array) -> jax.Array:
    """
    Weight ``1 - |x|_inf`` of the ``l^inf`` cone centred at the origin.

    Parameters
    ----------
    points : jax.Array
        Points, shape ``(..., 1, 2)``.

    Returns
    -------
    weight : jax.Array
        ``max(0, 1 - |x|_inf)``, shape ``(..., 1, 1)``.
    """
    return jnp.maximum(1.0 - jnp.max(jnp.abs(points), axis=-1, keepdims=True), 0.0)


@functools.cache
def compute_linf_poincare_constants(
    configuration: str, refinement: int = 32
) -> PoincareConstants:
    """
    Constants of (W2) and (W3) of a configuration of the ``l^inf`` family.

    Parameters
    ----------
    configuration : str
        ``"interior"``, ``"side"`` or ``"corner"``.
    refinement : int
        Number of mesh cells per unit length of the reference configuration,
        so the mesh size is ``s / refinement``.

    Returns
    -------
    constants : PoincareConstants
        ``c_2 h``, ``C_P h`` and ``h`` as multiples of the grid spacing
        ``s``.

    Notes
    -----
    The reference configurations at ``s = 1`` are the square ``(-1, 1)^2``,
    the rectangle ``(-1, 1) x (0, 1)`` with the Dirichlet side ``y = 0``, and
    the square ``(0, 1)^2`` with the Dirichlet sides ``x = 0`` and ``y = 0``,
    all with the weight ``1 - |x|_inf``; see
    ``compute_rectangle_poincare_constants``. The unweighted constants have
    the closed forms ``2 / pi``, ``2 / pi`` and ``sqrt(2) / pi``. Results are
    cached.
    """
    if configuration not in LINF_CONFIGURATIONS:
        raise ValueError(
            f"configuration must be one of {sorted(LINF_CONFIGURATIONS)}, got "
            f"{configuration!r}."
        )
    lower, upper, dirichlet_sides = LINF_CONFIGURATIONS[configuration]
    return compute_rectangle_poincare_constants(
        lower, upper, _linf_reference_weight, dirichlet_sides, refinement
    )


def round_up(value: float, decimals: int) -> float:
    """
    Round a positive number up to a number of decimals.

    Parameters
    ----------
    value : float
        Number to round.
    decimals : int
        Number of decimals kept.

    Returns
    -------
    rounded : float
        The smallest multiple of ``10^(-decimals)`` not below ``value``.
    """
    scale = 10.0**decimals
    return math.ceil(value * scale - 1e-9) / scale
