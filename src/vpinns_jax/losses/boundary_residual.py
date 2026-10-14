"""
Boundary penalty loss for trial functions without strong Dirichlet imposition.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax_fem.assembly import integrate_cellwise
from jax_fem.mesh import DfnMesh, TriangleMesh
from jax_fem.problem import ScalarCoefficient
from jax_fem.reference_cell import ReferenceInterval

from vpinns_jax.models.evaluation import Network, evaluate_network_values


def boundary_residual_loss(
    network: Network,
    mesh: TriangleMesh | DfnMesh,
    boundary_value: ScalarCoefficient,
    *,
    boundary_tags: tuple[int, ...] | None = None,
    quadrature_degree: int = 3,
) -> jax.Array:
    """
    Squared ``L^2`` norm of the boundary mismatch of a trial function.

    Parameters
    ----------
    network : Network
        Scalar trial function of one point.
    mesh : TriangleMesh | DfnMesh
        Mesh whose boundary facets carry the condition.
    boundary_value : ScalarCoefficient
        Dirichlet data ``g``.
    boundary_tags : tuple[int, ...] | None
        Boundary tags to include; every boundary facet if ``None``.
    quadrature_degree : int
        Exactness degree of the Gauss-Legendre rule on each facet.

    Returns
    -------
    loss : jax.Array
        ``int_Gamma (w - g)^2 ds``, shape ``()``.

    Notes
    -----
    This penalty is added to an interior loss for trial functions that do not
    satisfy the Dirichlet condition by construction. The facet selection
    reads the mesh tags on the host, so the function may be called inside a
    jitted loss that closes over the mesh.
    """
    quadrature = ReferenceInterval().create_quadrature(quadrature_degree)
    boundary_facets = np.asarray(mesh.boundary_facets)
    if boundary_tags is not None:
        selected = np.isin(np.asarray(mesh.boundary_facet_tags), boundary_tags)
        boundary_facets = boundary_facets[selected]

    facet_vertices = jnp.take(mesh.facets_to_vertices, boundary_facets, axis=-2)
    starts, ends = jnp.split(
        jnp.take(mesh.vertex_coordinates, facet_vertices, axis=-3), 2, axis=-3
    )
    edges = ends - starts
    points = starts + quadrature.points * edges
    facet_lengths = jnp.linalg.norm(edges, axis=-1, keepdims=True)
    mismatch = evaluate_network_values(network, points) - boundary_value(points)
    return jnp.sum(integrate_cellwise(mismatch**2, quadrature.weights * facet_lengths))
