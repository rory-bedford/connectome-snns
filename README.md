# Connectome-Constrained SNN Models

PyTorch-based research code modelling dynamical connectomics datasets with SNNs.


## Environment

We recommend [uv](https://docs.astral.sh/uv/) for dependency management.

You should manually sync your environment with either CPU or GPU support:

```bash
# For CPU
uv sync --extra cpu

# For GPU
uv sync --extra cu129
```

On the Zenke lab workstations, we need to store virtual environments on ereborfs. To do this we need to manually make a virtual environment and sync it. For example, I run:

```
uv venv /ereborfs/bedfrory/venvs/connectome-snns
source /ereborfs/bedford/venvs/connectome-snns/bin/activate
uv sync --active --extra cu129 --cache-dir /ereborfs/bedfrory/.cache/uv
```

Note the `active` flag makes uv install into the activated environment rather than the default in the project directory, and the `cache` flag makes uv use a non-default cache directory as our home quota on these machines is small.

## Development

For development purposes, we use a pre-commit hook that runs the [ruff](https://docs.astral.sh/ruff/) linter and formatter to ensure code quality and consistency before each commit.
For this to work please run `uv run pre-commit install` after cloning the repository.

## Notebooks

Interactive Jupyter notebooks demonstrating key experiments and analyses are available in the [`notebooks/`](notebooks/) directory. See the [notebooks README](notebooks/README.md) for an overview of available notebooks. Each notebook is also compiled to HTML using [Quarto](https://quarto.org/) for easy viewing without running code.

## Reproducibility

Special care is taken to make all our simulations fully reproducible. In particular, we perform full tracking of the exact code and parameters used to run each experiment, which are stored in metadata files alongside the experiment outputs.

To use these features, you need to write and run your python scripts as follows:

* Make sure your script has a `main(output_dir, params_csv)` function that accepts two arguments:
  - `output_dir`: Path to the directory where your experiment outputs should be saved
  - `params_csv`: Path to the CSV file containing your experiment parameters
* Please only use these paths for loading and saving data in order for our tracking to function properly
* `experiment.toml` in the repository root is a template you can use to configure your experiments
* We recommend you make a `workspace/` folder which will be gitignored, and copy this template in there alongside any parameter files you wish to modify on the fly
* Make sure the repository has all changes committed (files in `workspace/` are automatically ignored)
* You can then run your script using `./run` or `uv run python run_experiment.py`

Note the experiment running scripts can be passed a filepath to your `experiment.toml` file as input, and will use the editable version under `workspace/experiment.toml` by default.

With this system, we can perform full provenance tracking of our experiments and easily rerun them even years later.

For examples of usage please see `scripts/`.
