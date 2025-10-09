
# Copilot Instructions for Connectome-Constrained SNN Models

## Project Overview
- This project models dynamical connectomics datasets using Spiking Neural Networks (SNNs) in PyTorch.
- The `src/` directory contains reusable modules and utilities, which are imported and used by code in the `notebooks/` and `scripts/` folders.
- Utilities for graph construction and manipulation are in `src/utils/graph_builder.py`.
- Configuration files are in `config/`, scripts in `scripts/`, and research notebooks in `notebooks/`.


## Environment & Dependency Management
- Use [uv](https://docs.astral.sh/uv/) for Python dependency management.
- Sync environment for CPU or GPU support:
  - `uv sync --extra cpu` (CPU)
  - `uv sync --extra gpu` (GPU)
- Dependencies are tracked in `pyproject.toml` and `uv.lock`.
- Lint and format code using Ruff via uv:
  - `uv ruff check .` (lint)
  - `uv ruff format .` (format)



## Key Patterns & Conventions
- Follow modular design: keep data processing, model definition, and training logic in separate files.
- Use Google-style docstrings for all functions and classes to ensure research reproducibility and clarity.
- Static typing is used frequently throughout the codebase; annotate functions and variables with type hints.

## Workflows
- Notebooks in `notebooks/` are used for exploratory analysis and prototyping.
- Scripts in `scripts/` may automate data preparation, training, or evaluation.
- For debugging, use standard Python tools (e.g., `pdb`, VS Code debugger).

## Integration Points
- PyTorch is the primary ML framework; ensure compatibility with CPU/GPU as needed.
- Data flows from config/scripts/notebooks into model code in `src/`.
- External datasets or connectomics data should be referenced in config or scripts, not hardcoded.


## Examples
- To build a graph: see `src/utils/graph_builder.py` for construction patterns.
- To add a new model: create a new file in `src/`, register config options, and document usage in README or notebooks.
- When writing new functions or classes, use Google docstring format:
  ```python
  def example_function(param1: int, param2: str) -> bool:
    """
    Brief summary of the function.

    Args:
      param1 (int): Description of param1.
      param2 (str): Description of param2.

    Returns:
      bool: Description of the return value.
    """
    ...
  ```

## Additional Notes
- Keep code modular and well-documented for research clarity.
- Update README and this file when adding major features or changing workflows.

---

*Please provide feedback if any section is unclear or missing important project-specific details.*
