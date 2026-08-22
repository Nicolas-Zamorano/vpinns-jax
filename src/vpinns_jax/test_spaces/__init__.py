"""Test spaces: FE (Component 3, ``fe_space.py``) and mesh-free patch (``patch_space.py``)."""

from vpinns_jax.test_spaces.base import TestSpace
from vpinns_jax.test_spaces.fe_space import FETestSpace, create_fe_test_space
from vpinns_jax.test_spaces.patch_space import (
    PatchQuadrature,
    PatchTestSpace,
    create_patch_test_space,
    gather_patch_quadrature,
)

__all__ = [
    "TestSpace",
    "FETestSpace",
    "create_fe_test_space",
    "PatchTestSpace",
    "create_patch_test_space",
    "PatchQuadrature",
    "gather_patch_quadrature",
]
