"""
Variational physics-informed neural networks built on ``jax-fem``.

Submodules are imported directly; this package re-exports nothing. Importing
it imports ``jax_fem``, which enables 64-bit precision in JAX
(``jax_enable_x64``) before any array is created.
"""

import jax_fem  # noqa: F401

__all__: list[str] = []
