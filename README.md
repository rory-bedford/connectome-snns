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

For development purposes, we use a pre-commit hook that runs the [ruff](https://docs.astral.sh/ruff/) linter and formatter to ensure code quality and consistency before each commit.
For this to work please run `uv run pre-commit install` after cloning the repository.

## Notebooks

Interactive Jupyter notebooks demonstrating key experiments and analyses are available in the [`notebooks/`](notebooks/) directory. See the [notebooks README](notebooks/README.md) for an overview of available notebooks. Each notebook is also compiled to HTML using [Quarto](https://quarto.org/) for easy viewing without running code.
