"""
Create best_state folder by applying scaling factors from training to initial weights.

This script loads the initial network state from the initial_state folder and applies
the scaling factors from a specific epoch in training_metrics.csv to create an optimized
network state in a best_state folder.

Usage:
    python make_best_state.py <directory> <epoch>

Args:
    directory: Path to the training directory (containing initial_state, checkpoints, etc.)
    epoch: Epoch number to extract scaling factors from training_metrics.csv
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict
import sys
from datetime import datetime


def load_initial_state(initial_state_dir: Path) -> Dict[str, np.ndarray]:
    """
    Load the initial network structure from npz file.

    Args:
        initial_state_dir (Path): Directory containing network_structure.npz

    Returns:
        Dict[str, np.ndarray]: Dictionary with keys:
            - recurrent_weights: Initial recurrent weights
            - feedforward_weights: Initial feedforward weights
            - recurrent_connectivity: Connectivity mask for recurrent
            - feedforward_connectivity: Connectivity mask for feedforward
            - cell_type_indices: Cell type assignments for recurrent neurons
            - feedforward_cell_type_indices: Cell type assignments for input neurons
            - assembly_ids: Assembly assignments for recurrent neurons
    """
    npz_path = initial_state_dir / "network_structure.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Could not find network_structure.npz at {npz_path}")

    data = np.load(npz_path, allow_pickle=True)
    return {key: data[key] for key in data.files}


def load_scaling_factors_at_epoch(
    training_dir: Path, epoch: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load scaling factors for recurrent and feedforward connections at a specific epoch.

    Scaling factors are logged as flattened matrices in the CSV. They are indexed as:
    - Recurrent: scaling_factors/recurrent/source_type_target_type
    - Feedforward: scaling_factors/feedforward/source_type_target_type

    Args:
        training_dir (Path): Directory containing training_metrics.csv
        epoch (int): Epoch number to extract scaling factors for

    Returns:
        Tuple[np.ndarray, np.ndarray]: (scaling_factors, scaling_factors_FF)
            - scaling_factors: Shape (n_cell_types, n_cell_types) for recurrent
            - scaling_factors_FF: Shape (n_cell_types_FF, n_cell_types) for feedforward
    """
    csv_path = training_dir / "training_metrics.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Could not find training_metrics.csv at {csv_path}")

    # Load CSV
    df = pd.read_csv(csv_path)

    # Find the row for the requested epoch
    epoch_rows = df[df["epoch"] == epoch]
    if epoch_rows.empty:
        available_epochs = sorted(df["epoch"].unique())
        raise ValueError(
            f"Epoch {epoch} not found in training_metrics.csv. "
            f"Available epochs: {available_epochs}"
        )

    row = epoch_rows.iloc[0]

    # Extract all scaling factor columns
    recurrent_cols = sorted(
        [col for col in df.columns if col.startswith("scaling_factors/recurrent/")]
    )
    feedforward_cols = sorted(
        [col for col in df.columns if col.startswith("scaling_factors/feedforward/")]
    )

    if not recurrent_cols or not feedforward_cols:
        raise ValueError(
            f"Could not find scaling factor columns in training_metrics.csv. "
            f"Expected columns like 'scaling_factors/recurrent/...' and 'scaling_factors/feedforward/...'. "
            f"Found columns: {df.columns.tolist()}"
        )

    # Extract values for recurrent and feedforward
    recurrent_values = np.array([row[col] for col in recurrent_cols], dtype=np.float32)
    feedforward_values = np.array(
        [row[col] for col in feedforward_cols], dtype=np.float32
    )

    # Determine matrix dimensions from number of columns
    # For recurrent: n_cell_types^2 values flattened
    # For feedforward: n_cell_types_FF * n_cell_types values flattened
    n_recurrent_sq = len(recurrent_values)
    n_recurrent_types = int(np.sqrt(n_recurrent_sq))
    if n_recurrent_types**2 != n_recurrent_sq:
        raise ValueError(
            f"Cannot determine recurrent cell types from {n_recurrent_sq} scaling factor values. "
            f"Expected a perfect square number."
        )

    # Reshape into matrices
    scaling_factors = recurrent_values.reshape(n_recurrent_types, n_recurrent_types)

    # For feedforward, we need to know n_cell_types_FF (inferred from feedforward cols / recurrent types)
    n_feedforward_types = len(feedforward_values) // n_recurrent_types
    if n_feedforward_types * n_recurrent_types != len(feedforward_values):
        raise ValueError(
            f"Cannot determine feedforward dimensions from {len(feedforward_values)} values "
            f"and {n_recurrent_types} recurrent cell types."
        )

    scaling_factors_FF = feedforward_values.reshape(
        n_feedforward_types, n_recurrent_types
    )

    return (
        scaling_factors,
        scaling_factors_FF,
    )


def apply_scaling_to_weights(
    initial_weights: np.ndarray,
    scaling_factors: np.ndarray,
    cell_type_indices: np.ndarray,
    connectivity_mask: np.ndarray,
) -> np.ndarray:
    """
    Apply scaling factors to weights based on cell type.

    For each synapse (i→j), the weight is multiplied by scaling_factors[target_type, source_type],
    where target_type = cell_type_indices[i] and source_type = cell_type_indices[j].

    Args:
        initial_weights (np.ndarray): Initial weight matrix of shape (n_neurons, n_neurons)
        scaling_factors (np.ndarray): Scaling factors of shape (n_target_types, n_source_types)
        cell_type_indices (np.ndarray): Cell type assignment for each neuron
        connectivity_mask (np.ndarray): Connectivity mask (boolean)

    Returns:
        np.ndarray: Scaled weight matrix of same shape as initial_weights
    """
    scaled_weights = initial_weights.copy()
    n_neurons = len(cell_type_indices)

    # Apply scaling: weight[i,j] *= scaling_factors[cell_type[i], cell_type[j]]
    for i in range(n_neurons):
        for j in range(n_neurons):
            if connectivity_mask[i, j]:
                target_type = cell_type_indices[i]
                source_type = cell_type_indices[j]
                scale = scaling_factors[target_type, source_type]
                scaled_weights[i, j] *= scale

    return scaled_weights


def create_best_state_folder(
    training_dir: Path, epoch: int, best_state_dir: Path
) -> None:
    """
    Create best_state folder with scaled weights from a specific epoch.

    Args:
        training_dir (Path): Directory containing initial_state and training_metrics.csv
        epoch (int): Epoch number to extract scaling factors from
        best_state_dir (Path): Output directory for best_state
    """
    print(f"Loading initial state from {training_dir / 'initial_state'}...")
    initial_state = load_initial_state(training_dir / "initial_state")

    print(f"Loading scaling factors from epoch {epoch}...")
    scaling_factors, scaling_factors_FF = load_scaling_factors_at_epoch(
        training_dir, epoch
    )

    # Apply scaling to recurrent weights
    print("Applying scaling factors to recurrent weights...")
    scaled_recurrent_weights = apply_scaling_to_weights(
        initial_weights=initial_state["recurrent_weights"],
        scaling_factors=scaling_factors,
        cell_type_indices=initial_state["cell_type_indices"],
        connectivity_mask=initial_state["recurrent_connectivity"],
    )

    # Apply scaling to feedforward weights
    print("Applying scaling factors to feedforward weights...")
    scaled_feedforward_weights = apply_scaling_to_weights(
        initial_weights=initial_state["feedforward_weights"],
        scaling_factors=scaling_factors_FF,
        cell_type_indices=initial_state["feedforward_cell_type_indices"],
        connectivity_mask=initial_state["feedforward_connectivity"],
    )

    # Create best_state directory
    best_state_dir.mkdir(parents=True, exist_ok=True)

    # Save the scaled network structure
    print(f"Saving best state to {best_state_dir}...")
    np.savez(
        best_state_dir / "network_structure.npz",
        recurrent_weights=scaled_recurrent_weights,
        feedforward_weights=scaled_feedforward_weights,
        recurrent_connectivity=initial_state["recurrent_connectivity"],
        feedforward_connectivity=initial_state["feedforward_connectivity"],
        cell_type_indices=initial_state["cell_type_indices"],
        feedforward_cell_type_indices=initial_state["feedforward_cell_type_indices"],
        assembly_ids=initial_state["assembly_ids"],
    )

    print(f"✓ Best state saved to {best_state_dir / 'network_structure.npz'}")

    # Create README documenting the epoch and scaling factors
    readme_path = best_state_dir / "README.md"
    readme_content = f"""# Best State Network

This folder contains the network state with scaling factors applied from training epoch {epoch}.

## Creation Details

- **Epoch**: {epoch}
- **Created**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
- **Source**: Initial weights scaled by training metrics from epoch {epoch}

## Contents

- `network_structure.npz`: Network structure with scaled weights
  - `recurrent_weights`: Recurrent connection weights with scaling applied
  - `feedforward_weights`: Feedforward connection weights with scaling applied
  - `recurrent_connectivity`: Connectivity mask for recurrent connections
  - `feedforward_connectivity`: Connectivity mask for feedforward connections
  - `cell_type_indices`: Cell type assignments for recurrent neurons
  - `feedforward_cell_type_indices`: Cell type assignments for input neurons
  - `assembly_ids`: Assembly assignments for recurrent neurons

## Scaling Factors Applied

### Recurrent Scaling Factors (shape: {scaling_factors.shape})
```
{scaling_factors}
```

### Feedforward Scaling Factors (shape: {scaling_factors_FF.shape})
```
{scaling_factors_FF}
```

## Usage

Load the network structure with:
```python
import numpy as np
data = np.load('network_structure.npz', allow_pickle=True)
recurrent_weights = data['recurrent_weights']
feedforward_weights = data['feedforward_weights']
# ... etc
```
"""

    with open(readme_path, "w") as f:
        f.write(readme_content)

    print(f"✓ README created at {readme_path}")


def main():
    """Parse arguments and run the best_state creation."""
    parser = argparse.ArgumentParser(
        description="Create best_state folder by applying scaling factors from training."
    )
    parser.add_argument(
        "directory",
        type=Path,
        help="Directory containing initial_state and training_metrics.csv",
    )
    parser.add_argument(
        "epoch",
        type=int,
        help="Epoch number to extract scaling factors from",
    )

    args = parser.parse_args()

    # Validate inputs
    if not args.directory.exists():
        print(f"Error: Directory does not exist: {args.directory}", file=sys.stderr)
        sys.exit(1)

    if not (args.directory / "initial_state").exists():
        print(
            f"Error: initial_state folder not found in {args.directory}",
            file=sys.stderr,
        )
        sys.exit(1)

    if not (args.directory / "training_metrics.csv").exists():
        print(
            f"Error: training_metrics.csv not found in {args.directory}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Create best_state directory path
    best_state_dir = args.directory / "best_state"

    try:
        create_best_state_folder(args.directory, args.epoch, best_state_dir)
        print(f"\n✓ Successfully created best_state folder from epoch {args.epoch}")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
