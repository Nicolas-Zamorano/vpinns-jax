# `vpinns-jax`

Variational Physics-Informed Neural Networks (VPINNs) and Mesh-Free Variational PINNs (MF-VPINNs) built on [JAX](https://github.com/google/jax), [Equinox](https://github.com/patrick-kidger/equinox), [Optax](https://github.com/google-deepmind/optax), and [`jax-fem`](/home/nicolas/Documents/Github/fem-jax).

---

## Documentation

For a comprehensive explanation of the repository architecture, mathematical theory, modules, examples, and migration plan, see:
👉 **[REPOSITORY_OVERVIEW.md](REPOSITORY_OVERVIEW.md)**

---

## Quickstart

### Installation & Environment
This project uses [`uv`](https://docs.astral.sh/uv/) for environment and dependency management.

```bash
# Sync dependencies
uv sync
```

### Running Tests
```bash
uv run pytest
```

### Running Examples
```bash
# 01. Direct trial with FE test space (Euclidean vs. Riesz loss)
uv run python examples/01_vpinns_poisson_direct.py

# 02. Interpolated trial with FE test space
uv run python examples/02_vpinns_interpolated.py

# 03. Mesh-Free VPINN with Shepard-normalised patch space
uv run python examples/03_mf_vpinns_patch.py

# 04. Mathematical verification of Shepard normalisation (no training)
uv run python examples/04_shepard_normalisation.py

# 05. Uniform convergence study under patch refinement
uv run python examples/05_convergence_study.py

# 06. Discrete Fracture Network (DFN) with interpolated trial
uv run python examples/06_dfn_interpolated.py
```
Output figures, convergence logs, and error metrics are stored in timestamped folders in `experiments/`.