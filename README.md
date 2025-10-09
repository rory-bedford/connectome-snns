# Connectome-Constrained SNN Models

PyTorch-based research code modelling dynamical connectomics datasets with SNNs.


## Environment

We recommend [uv](https://docs.astral.sh/uv/) for dependency management.

You should manually sync your environment with either CPU or GPU support:

```bash
# For CPU
uv sync --extra cpu

# For GPU
uv sync --extra gpu
```

## Development

For development purposes, we use a pre-commit hook that runs the [uv ruff](https://docs.astral.sh/uv/) linter and formatter to ensure code quality and consistency before each commit.
