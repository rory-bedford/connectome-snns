"""Cell type assignment utilities.

This module provides functions for assigning neurons to different cell types
based on given proportions.

Features:
- Random assignment based on proportions
- Deterministic seeding for reproducibility
- Support for multiple cell types

Typical workflow:
    >>> cell_proportions = [0.8, 0.2]  # 80% type 0, 20% type 1
    >>> cell_types = assign_cell_types(100, cell_proportions, seed=42)
    >>> # cell_types contains integers 0 and 1 indicating cell type
"""

import numpy as np
from numpy.typing import NDArray
from typing import List

# Type aliases for clarity
IntArray = NDArray[np.int_]


def assign_cell_types(
    num_neurons: int, 
    cell_type_proportions: List[float], 
) -> IntArray:
    """
    Randomly assign neurons to cell types based on given proportions.
    
    Args:
        num_neurons (int): Total number of neurons to assign.
        cell_type_proportions (List[float]): List of proportions for each cell type. Must sum to 1.0.
        
    Returns:
        IntArray: Array of shape (num_neurons,) with cell type indices.
    
    Raises:
        ValueError: If proportions don't sum to approximately 1.0.
    """
    if abs(sum(cell_type_proportions) - 1.0) > 1e-6:
        raise ValueError(f"Cell type proportions must sum to 1.0, got {sum(cell_type_proportions)}")
    
    # Calculate number of neurons for each cell type
    num_cell_types = len(cell_type_proportions)
    cell_type_counts = np.zeros(num_cell_types, dtype=int)
    
    # Assign neurons ensuring we use all neurons
    remaining_neurons = num_neurons
    for i in range(num_cell_types - 1):
        count = int(np.round(num_neurons * cell_type_proportions[i]))
        cell_type_counts[i] = count
        remaining_neurons -= count
    
    # Last cell type gets remaining neurons
    cell_type_counts[-1] = remaining_neurons
    
    # Create array of cell type assignments
    cell_type_indices = np.zeros(num_neurons, dtype=int)
    start_idx = 0
    
    for cell_type_idx, count in enumerate(cell_type_counts):
        cell_type_indices[start_idx:start_idx + count] = cell_type_idx
        start_idx += count
    
    # Shuffle to randomize assignment
    np.random.shuffle(cell_type_indices)
    
    return cell_type_indices