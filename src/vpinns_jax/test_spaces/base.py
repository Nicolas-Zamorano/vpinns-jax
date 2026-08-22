"""``TestSpace`` protocol: the common contract ``losses``/``estimators`` code against.

FE test spaces (``fe_space.py``) and mesh-free patch test spaces
(``patch_space.py``) have genuinely different internal representations (a
global sparse-assembled space vs. a per-patch local quadrature gather), so
they are not implemented behind one shared base class. This module documents
the attributes both share conceptually (implementation plan Section 3),
useful as a typing/documentation anchor; concrete residual/loss code
dispatches on the concrete type.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import jax


@runtime_checkable
class TestSpace(Protocol):
    """Common attributes of a discrete test space ``V_test = span{psi_i}``.

    Attributes
    ----------
    number_of_test_functions : int
        ``|I_test|``.
    """

    number_of_test_functions: int


def diameter_scaling(diameters: jax.Array) -> jax.Array:
    """``h_i``, unchanged: the per-test-function length scale used by
    ``estimators.aposteriori`` (patch diameter for a patch space, cell
    diameter for an FE space). Kept as a one-line named indirection so
    callers document *which* length scale they mean.
    """
    return diameters
