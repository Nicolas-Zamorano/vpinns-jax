"""Weak boundary residual loss ``L_Gamma(w_theta) = int_Gamma (w_theta - g)^2 ds``.

**Scoping decision.** The math reference's weak boundary formulation
(Section 3) is a mixed ``H(div)`` dual pairing, ``r_Gamma(w, q) = int_Gamma
(w - g)(q . n) ds`` tested against an ``H(div)`` flux space -- a
genuinely mixed-formulation construction. This module implements the more
common, practical simplification used across the PINN/VPINN literature: a
direct ``L^2(Gamma)`` boundary penalty, evaluated by 1D Gauss-Legendre
quadrature along the mesh's boundary edges. It is a *loss term added to* the
interior residual loss (Euclidean/Riesz), for a trial function that does
*not* already satisfy the boundary condition exactly -- if it does (a
``models.boundary.ConstrainedAnsatz`` direct trial, or an interpolated trial
with boundary DOFs set to ``g``), no boundary loss term is needed at all.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax_fem.mesh import TriangleMesh
from jax_fem.reference_cell import ReferenceInterval

from vpinns_jax.pde.base import ScalarCoefficient

__all__ = ["boundary_residual_loss"]


def boundary_residual_loss(
    network,
    mesh: TriangleMesh,
    boundary_value: ScalarCoefficient,
    *,
    boundary_tags: tuple[int, ...] | None = None,
    quadrature_degree: int = 3,
) -> jax.Array:
    """``L_Gamma(w_theta) = int_Gamma (w_theta - g)^2 ds``.

    Parameters
    ----------
    network : Callable[[jax.Array], jax.Array]
        Single-point scalar trial evaluator.
    mesh : TriangleMesh
    boundary_value : ScalarCoefficient
        ``g(x)``.
    boundary_tags : tuple[int, ...] | None
        Restrict to these boundary tags; defaults to the whole boundary.
    quadrature_degree : int
        1D Gauss-Legendre exactness degree along each boundary edge.

    Returns
    -------
    jax.Array
        Scalar, shape ``()``.
    """
    quadrature = ReferenceInterval().create_quadrature(quadrature_degree)
    parameter = quadrature.points[0, :, 0, 0]  # (Q,), in [0, 1]
    weight = quadrature.weights[0, :, 0, 0]  # (Q,), sums to 1

    boundary_facets = mesh.boundary_facets
    if boundary_tags is not None:
        selected = jnp.isin(mesh.boundary_facet_tags, jnp.asarray(boundary_tags))
        boundary_facets = boundary_facets[selected]

    facet_vertices = mesh.facets_to_vertices[boundary_facets]  # (F, 2)
    vertex_a = mesh.vertex_coordinates[facet_vertices[:, 0], 0, :]  # (F, 2)
    vertex_b = mesh.vertex_coordinates[facet_vertices[:, 1], 0, :]  # (F, 2)
    edge_vector = vertex_b - vertex_a
    facet_lengths = jnp.linalg.norm(edge_vector, axis=-1)  # (F,)

    points = vertex_a[:, None, :] + parameter[None, :, None] * edge_vector[:, None, :]  # (F,Q,2)
    values = jax.vmap(jax.vmap(network))(points)  # (F, Q)
    target = boundary_value(points[..., None, :])[..., 0, 0]  # (F, Q)

    integrand = (values - target) ** 2
    per_facet = jnp.sum(weight[None, :] * integrand, axis=-1) * facet_lengths  # (F,)
    return jnp.sum(per_facet)
