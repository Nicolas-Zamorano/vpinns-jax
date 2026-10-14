"""
Riesz losses: dual norms of residual vectors.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.scipy.linalg
from jax_fem.assembly import GlobalMatrix


def restrict_to_free(gram: GlobalMatrix, free_indices: jax.Array) -> jax.Array:
    """
    Dense restriction of a global Gram matrix to the free indices.

    Parameters
    ----------
    gram : GlobalMatrix
        Gram matrix, shape ``(dofs_test, dofs_test)``.
    free_indices : jax.Array
        Indices of the test functions, shape ``(N_free,)``.

    Returns
    -------
    restricted_gram : jax.Array
        Dense Gram matrix of the free test functions, shape
        ``(N_free, N_free)``.

    Notes
    -----
    The restriction is dense because the Riesz loss needs a factorization of
    it, computed once with ``jnp.linalg.cholesky``. This limits the number of
    free test functions to a few thousand.
    """
    dense = gram.todense()
    return jnp.take(jnp.take(dense, free_indices, axis=-2), free_indices, axis=-1)


def riesz_loss(free_residual: jax.Array, factor: jax.Array) -> jax.Array:
    """
    Squared dual norm of a residual vector.

    Parameters
    ----------
    free_residual : jax.Array
        Residual at the free test functions, shape ``(N_free, 1)``.
    factor : jax.Array
        Lower Cholesky factor ``L`` of the restricted Gram matrix
        ``G = L L^T``, shape ``(N_free, N_free)``.

    Returns
    -------
    loss : jax.Array
        ``r^T G^{-1} r = |L^{-1} r|^2``, shape ``()``.

    Notes
    -----
    ``r^T G^{-1} r`` is the squared dual norm of the residual over the span
    of the test functions, measured in the energy norm. The factor depends
    only on the test space, so it is computed once before training; each
    evaluation costs one triangular solve.
    """
    whitened = jax.scipy.linalg.solve_triangular(factor, free_residual, lower=True)
    return jnp.sum(whitened**2)


def diagonal_riesz_loss(
    free_residual: jax.Array, gram_diagonal: jax.Array
) -> jax.Array:
    """
    Decoupled squared dual norm of a residual vector.

    Parameters
    ----------
    free_residual : jax.Array
        Residual at the free test functions, shape ``(N_free, 1)``.
    gram_diagonal : jax.Array
        Diagonal of the restricted Gram matrix, shape ``(N_free,)``.

    Returns
    -------
    loss : jax.Array
        ``r^T D^{-1} r`` with ``D = diag(G)``, shape ``()``.

    Notes
    -----
    Each test function is treated as if it spanned a space of its own. The
    result is not equivalent to ``riesz_loss`` uniformly in the mesh size:
    for smooth residual modes the two differ by a factor up to ``h^-2``.
    """
    return jnp.sum(free_residual**2 / jnp.expand_dims(gram_diagonal, axis=-1))
