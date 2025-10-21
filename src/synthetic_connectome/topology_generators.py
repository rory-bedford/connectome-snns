"""Generate graph topologies with cell type assignments.

This module provides functions to generate boolean adjacency matrices with
cell type assignments based on connectivity probability matrices between cell types.

Features:
- Multiple cell types (not limited to excitatory/inhibitory)
- Connectivity probability matrices between cell types
- Boolean (unsigned) adjacency matrices
- Cell type indices based on position in configuration

Typical workflow:
    >>> # Matrix-based connectivity with multiple cell types
    >>> conn_probs = [[0.05, 0.05], [0.3, 0.05]]  # 2x2 matrix for 2 cell types
    >>> cell_proportions = [0.8, 0.2]  # 80% type 0, 20% type 1
    >>> adj, cell_indices = sparse_graph_generator(
    ...     num_neurons=100, 
    ...     conn_matrix=conn_probs, 
    ...     cell_type_proportions=cell_proportions
    ... )
    >>> # adj is a boolean adjacency matrix
    >>> # cell_indices contains 0s and 1s indicating cell type
"""

import numpy as np
from numpy.typing import NDArray
from typing import Union, List

# Type aliases for clarity
IntArray = NDArray[np.int_]
BoolArray = NDArray[np.bool_]
FloatArray = NDArray[np.float64]


def assign_cell_types(
    num_neurons: int, 
    cell_type_proportions: List[float], 
    seed: int | None = None
) -> IntArray:
    """
    Randomly assign neurons to cell types based on given proportions.
    
    Args:
        num_neurons (int): Total number of neurons to assign.
        cell_type_proportions (List[float]): List of proportions for each cell type. Must sum to 1.0.
        seed (int | None): Random seed for reproducibility.
        
    Returns:
        IntArray: Array of shape (num_neurons,) with cell type indices.
    
    Raises:
        ValueError: If proportions don't sum to approximately 1.0.
    """
    if abs(sum(cell_type_proportions) - 1.0) > 1e-6:
        raise ValueError(f"Cell type proportions must sum to 1.0, got {sum(cell_type_proportions)}")
    
    rng = np.random.default_rng(seed)
    
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
    rng.shuffle(cell_type_indices)
    
    return cell_type_indices


def sparse_graph_generator(
    num_neurons: int | tuple[int, int],
    conn_matrix: List[List[float]],
    cell_type_proportions: List[float],
    seed: int | None = None,
) -> tuple[BoolArray, IntArray]:
    """
    Generate a sparse random graph using the Erdos-Renyi model with cell types.

    Creates boolean adjacency matrix based on connectivity probability matrix
    between different cell types.

    Args:
        num_neurons (int | tuple[int, int]): Number of nodes in the graph.
        conn_matrix (List[List[float]]): NxN matrix of connection probabilities between cell types.
        cell_type_proportions (List[float]): List of proportions for each cell type.
        seed (int | None): Random seed for reproducibility.

    Returns:
        tuple[BoolArray, IntArray]: Boolean adjacency matrix and cell type indices (0, 1, 2, ...).
    """
    rng = np.random.default_rng(seed)

    # Parse num_neurons into source and target dimensions
    if isinstance(num_neurons, int):
        n_source = n_target = num_neurons
    else:
        n_source, n_target = num_neurons

    # Validate input parameters
    conn_matrix_np = np.array(conn_matrix)
    num_cell_types = len(cell_type_proportions)
    
    if conn_matrix_np.shape != (num_cell_types, num_cell_types):
        raise ValueError(f"conn_matrix shape {conn_matrix_np.shape} doesn't match "
                       f"number of cell types {num_cell_types}")
    
    # Assign cell types to source neurons
    source_cell_types = assign_cell_types(n_source, cell_type_proportions, seed)
    
    # For feedforward connections, assign target cell types too
    if n_source != n_target:
        target_cell_types = assign_cell_types(n_target, cell_type_proportions, seed)
    else:
        target_cell_types = source_cell_types
    
    # Generate boolean adjacency matrix based on cell type connectivity
    adjacency = np.zeros((n_source, n_target), dtype=bool)
    
    for i in range(n_source):
        for j in range(n_target):
            if n_source == n_target and i == j:
                continue  # No self-loops
            
            source_type = source_cell_types[i]
            target_type = target_cell_types[j]
            conn_prob = conn_matrix_np[source_type, target_type]
            
            if rng.random() < conn_prob:
                adjacency[i, j] = True
    
    return adjacency, source_cell_types


def generate_cross_layer_connectivity(
    n_source: int,
    n_target: int,
    source_cell_proportions: List[float],
    target_cell_indices: IntArray,
    conn_matrix: List[List[float]],
    seed: int | None = None,
) -> BoolArray:
    """
    Generate connectivity between two layers with different cell type structures.
    
    This function creates connections between a source layer (e.g., input mitral cells)
    and a target layer (e.g., connectome excitatory/inhibitory neurons) where the
    cell type structures may differ between layers.
    
    Args:
        n_source (int): Number of neurons in source layer.
        n_target (int): Number of neurons in target layer.
        source_cell_proportions (List[float]): Proportions of each cell type in source layer.
        target_cell_indices (IntArray): Cell type assignments for target layer neurons.
        conn_matrix (List[List[float]]): Matrix where conn_matrix[i][j] is probability from source type i to target type j.
        seed (int | None): Random seed for reproducibility.
        
    Returns:
        BoolArray: Boolean adjacency matrix of shape (n_source, n_target).
    """
    rng = np.random.default_rng(seed)
    
    # Assign cell types to source neurons
    source_cell_indices = assign_cell_types(n_source, source_cell_proportions, seed)
    
    conn_matrix_np = np.array(conn_matrix)
    num_source_types, num_target_types = conn_matrix_np.shape
    
    # Validate dimensions
    if len(source_cell_proportions) != num_source_types:
        raise ValueError(f"source_cell_proportions length {len(source_cell_proportions)} "
                        f"doesn't match conn_matrix rows {num_source_types}")
    
    if target_cell_indices.max() >= num_target_types:
        raise ValueError(f"target_cell_indices max {target_cell_indices.max()} "
                        f"exceeds conn_matrix columns {num_target_types}")
    
    # Generate connectivity matrix
    adjacency = np.zeros((n_source, n_target), dtype=bool)
    
    for i in range(n_source):
        for j in range(n_target):
            source_type = source_cell_indices[i]
            target_type = target_cell_indices[j]
            conn_prob = conn_matrix_np[source_type, target_type]
            
            if rng.random() < conn_prob:
                adjacency[i, j] = True
    
    return adjacency


def dense_graph_generator(
    num_neurons: int | tuple[int, int],
    cell_type_proportions: List[float],
    seed: int | None = None,
) -> tuple[BoolArray, IntArray]:
    """
    Generate a fully connected (dense) graph with cell types.

    All possible edges exist except self-loops (in square matrices). 
    Cell types are assigned based on given proportions.

    Args:
        num_neurons (int | tuple[int, int]): Number of nodes in the graph.
        cell_type_proportions (List[float]): List of proportions for each cell type.
        seed (int | None): Random seed for reproducibility.

    Returns:
        tuple[BoolArray, IntArray]: Boolean adjacency matrix and cell type indices (0, 1, 2, ...).
    """
    # Parse num_neurons into source and target dimensions
    if isinstance(num_neurons, int):
        n_source = n_target = num_neurons
    else:
        n_source, n_target = num_neurons

    # Generate fully connected adjacency matrix
    adjacency = np.ones((n_source, n_target), dtype=bool)

    # No self-loops only for square matrices
    if n_source == n_target:
        np.fill_diagonal(adjacency, 0)

    # Assign cell types to source neurons
    source_cell_types = assign_cell_types(n_source, cell_type_proportions, seed)
    
    return adjacency, source_cell_types


def assembly_generator(
    num_neurons: int | tuple[int, int],
    num_assemblies: int,
    conn_within: List[List[float]],
    conn_between: List[List[float]],
    cell_type_proportions: List[float],
    seed: int | None = None,
) -> tuple[BoolArray, IntArray]:
    """
    Generate a graph with assembly structure and cell types.

    Creates boolean adjacency matrix with assembly structure using 
    connectivity probability matrices between different cell types.

    Args:
        num_neurons (int | tuple[int, int]): Number of nodes in the graph.
        num_assemblies (int): Number of assemblies to create.
        conn_within (List[List[float]]): NxN matrix of connection probabilities within assemblies between cell types.
        conn_between (List[List[float]]): NxN matrix of connection probabilities between assemblies between cell types.
        cell_type_proportions (List[float]): List of proportions for each cell type.
        seed (int | None): Random seed for reproducibility.

    Returns:
        tuple[BoolArray, IntArray]: Boolean adjacency matrix and cell type indices (0, 1, 2, ...).
    """
    # Parse num_neurons into source and target dimensions
    if isinstance(num_neurons, int):
        n_source = n_target = num_neurons
    else:
        n_source, n_target = num_neurons

    if num_assemblies > n_source:
        raise ValueError("Number of assemblies cannot exceed number of source nodes.")

    rng = np.random.default_rng(seed)

    # Validate input parameters
    conn_within_np = np.array(conn_within)
    conn_between_np = np.array(conn_between)
    num_cell_types = len(cell_type_proportions)
    
    if conn_within_np.shape != (num_cell_types, num_cell_types):
        raise ValueError(f"conn_within shape {conn_within_np.shape} doesn't match "
                       f"number of cell types {num_cell_types}")
    if conn_between_np.shape != (num_cell_types, num_cell_types):
        raise ValueError(f"conn_between shape {conn_between_np.shape} doesn't match "
                       f"number of cell types {num_cell_types}")
    
    # Assign cell types to source neurons
    source_cell_types = assign_cell_types(n_source, cell_type_proportions, seed)
    
    # For feedforward connections, assign target cell types too
    if n_source != n_target:
        target_cell_types = assign_cell_types(n_target, cell_type_proportions, seed)
    else:
        target_cell_types = source_cell_types
    
    # Assign source and target nodes to assemblies (roughly equal sizes)
    assembly_assignments_source = np.array_split(np.arange(n_source), num_assemblies)
    assembly_assignments_target = np.array_split(np.arange(n_target), num_assemblies)

    # Create within-assembly mask using outer products
    within_assembly_mask = np.zeros((n_source, n_target), dtype=bool)
    for source_assembly, target_assembly in zip(
        assembly_assignments_source, assembly_assignments_target
    ):
        # Create indicator vectors for this assembly
        source_in_assembly = np.isin(np.arange(n_source), source_assembly)
        target_in_assembly = np.isin(np.arange(n_target), target_assembly)

        # Outer product gives True where both source and target are in this assembly
        within_assembly_mask |= np.outer(source_in_assembly, target_in_assembly)

    # Generate boolean adjacency matrix based on assembly structure and cell type connectivity
    adjacency = np.zeros((n_source, n_target), dtype=bool)
    
    for i in range(n_source):
        for j in range(n_target):
            if n_source == n_target and i == j:
                continue  # No self-loops
            
            source_type = source_cell_types[i]
            target_type = target_cell_types[j]
            
            # Choose connection probability based on assembly membership
            if within_assembly_mask[i, j]:
                conn_prob = conn_within_np[source_type, target_type]
            else:
                conn_prob = conn_between_np[source_type, target_type]
            
            if rng.random() < conn_prob:
                adjacency[i, j] = True
    
    return adjacency, source_cell_types
