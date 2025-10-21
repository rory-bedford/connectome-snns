"""Generate graph topologies with cell type assignments.

This module provides functions to generate boolean adjacency matrices with
cell type assignments based on connectivity probability matrices between cell types.

Features:
- Multiple cell types (not limited to excitatory/inhibitory)
- Connectivity probability matrices between cell types
- Boolean (unsigned) adjacency matrices
- Cell type indices based on position in configuration

Typical workflow:
    >>> # First assign cell types
    >>> cell_proportions = [0.8, 0.2]  # 80% type 0, 20% type 1
    >>> source_cell_types = assign_cell_types(100, cell_proportions, seed=42)
    >>> target_cell_types = assign_cell_types(100, cell_proportions, seed=43)
    >>>
    >>> # Then generate connectivity with matrix-based probabilities
    >>> conn_probs = [[0.05, 0.05], [0.3, 0.05]]  # 2x2 matrix for 2 cell types
    >>> adj = sparse_graph_generator(
    ...     source_cell_types, target_cell_types, conn_probs
    ... )
    >>> # adj is a boolean adjacency matrix
"""

import numpy as np
from numpy.typing import NDArray
from typing import List

# Type aliases for clarity
IntArray = NDArray[np.int_]
BoolArray = NDArray[np.bool_]
FloatArray = NDArray[np.float64]


def sparse_graph_generator(
    source_cell_types: IntArray,
    target_cell_types: IntArray,
    conn_matrix: List[List[float]],
    allow_self_loops: bool = False,
) -> BoolArray:
    """
    Generate a sparse random graph using the Erdos-Renyi model with cell types.

    Creates boolean adjacency matrix based on connectivity probability matrix
    between different cell types. Supports both recurrent and feedforward connections.

    Args:
        source_cell_types (IntArray): Cell type indices for source neurons.
        target_cell_types (IntArray): Cell type indices for target neurons.
        conn_matrix (List[List[float]]): Matrix of connection probabilities from source to target cell types.
        allow_self_loops (bool): Whether to allow connections from neuron i to itself.

    Returns:
        BoolArray: Boolean adjacency matrix of shape (len(source_cell_types), len(target_cell_types)).

    Note:
        For recurrent connections, pass the same array for both source_cell_types and target_cell_types.
    """
    n_source = len(source_cell_types)
    n_target = len(target_cell_types)

    conn_matrix_np = np.array(conn_matrix)
    n_source_types, n_target_types = conn_matrix_np.shape

    # Basic assertions
    assert source_cell_types.max() < n_source_types, (
        f"source_cell_types max {source_cell_types.max()} exceeds conn_matrix rows {n_source_types}"
    )
    assert target_cell_types.max() < n_target_types, (
        f"target_cell_types max {target_cell_types.max()} exceeds conn_matrix columns {n_target_types}"
    )

    # Generate connectivity
    adjacency = np.zeros((n_source, n_target), dtype=bool)

    for i in range(n_source):
        for j in range(n_target):
            # Skip self-loops if not allowed
            if not allow_self_loops and n_source == n_target and i == j:
                continue

            source_type = source_cell_types[i]
            target_type = target_cell_types[j]
            conn_prob = conn_matrix_np[source_type, target_type]

            if np.random.random() < conn_prob:
                adjacency[i, j] = True

    return adjacency


def dense_graph_generator(
    source_cell_types: IntArray,
    target_cell_types: IntArray,
    allow_self_loops: bool = False,
) -> BoolArray:
    """
    Generate a fully connected (dense) graph with cell types.

    All possible edges exist except self-loops (unless allow_self_loops=True).
    This is implemented as calling sparse_graph_generator with probability 1.0.

    Args:
        source_cell_types (IntArray): Cell type indices for source neurons.
        target_cell_types (IntArray): Cell type indices for target neurons.
        allow_self_loops (bool): Whether to allow connections from neuron i to itself.

    Returns:
        BoolArray: Boolean adjacency matrix of shape (len(source_cell_types), len(target_cell_types)).

    Note:
        For recurrent connections, pass the same array for both source_cell_types and target_cell_types.
    """
    # Create all-ones connectivity matrix based on cell types present
    n_source_types = source_cell_types.max() + 1
    n_target_types = target_cell_types.max() + 1
    conn_matrix = [[1.0] * n_target_types for _ in range(n_source_types)]

    return sparse_graph_generator(
        source_cell_types, target_cell_types, conn_matrix, allow_self_loops
    )


def assembly_generator(
    source_cell_types: IntArray,
    target_cell_types: IntArray,
    num_assemblies: int,
    conn_within: List[List[float]],
    conn_between: List[List[float]],
    allow_self_loops: bool = False,
) -> BoolArray:
    """
    Generate a graph with assembly structure and cell types.

    Creates boolean adjacency matrix with assembly structure using
    connectivity probability matrices between different cell types.

    Args:
        source_cell_types (IntArray): Cell type indices for source neurons.
        target_cell_types (IntArray): Cell type indices for target neurons.
        num_assemblies (int): Number of assemblies to create.
        conn_within (List[List[float]]): Matrix of connection probabilities within assemblies between cell types.
        conn_between (List[List[float]]): Matrix of connection probabilities between assemblies between cell types.
        allow_self_loops (bool): Whether to allow connections from neuron i to itself.

    Returns:
        BoolArray: Boolean adjacency matrix of shape (len(source_cell_types), len(target_cell_types)).

    Note:
        For recurrent connections, pass the same array for both source_cell_types and target_cell_types.
    """
    n_source = len(source_cell_types)
    n_target = len(target_cell_types)

    # Basic assertions
    assert num_assemblies <= min(n_source, n_target), (
        "Number of assemblies cannot exceed min(n_source, n_target)"
    )

    conn_within_np = np.array(conn_within)
    conn_between_np = np.array(conn_between)
    n_source_types, n_target_types = conn_within_np.shape

    assert conn_within_np.shape == conn_between_np.shape, (
        "conn_within and conn_between must have same shape"
    )
    assert source_cell_types.max() < n_source_types, (
        f"source_cell_types max {source_cell_types.max()} exceeds conn_within rows {n_source_types}"
    )
    assert target_cell_types.max() < n_target_types, (
        f"target_cell_types max {target_cell_types.max()} exceeds conn_within columns {n_target_types}"
    )

    # Create assembly assignments
    assembly_assignments_source = np.array_split(np.arange(n_source), num_assemblies)
    assembly_assignments_target = np.array_split(np.arange(n_target), num_assemblies)

    # Create within-assembly mask
    within_assembly_mask = np.zeros((n_source, n_target), dtype=bool)
    for source_assembly, target_assembly in zip(
        assembly_assignments_source, assembly_assignments_target
    ):
        source_in_assembly = np.isin(np.arange(n_source), source_assembly)
        target_in_assembly = np.isin(np.arange(n_target), target_assembly)
        within_assembly_mask |= np.outer(source_in_assembly, target_in_assembly)

    # Generate connectivity with assembly-dependent probabilities
    adjacency = np.zeros((n_source, n_target), dtype=bool)

    for i in range(n_source):
        for j in range(n_target):
            # Skip self-loops if not allowed
            if not allow_self_loops and n_source == n_target and i == j:
                continue

            source_type = source_cell_types[i]
            target_type = target_cell_types[j]

            # Choose connection probability based on assembly membership
            if within_assembly_mask[i, j]:
                conn_prob = conn_within_np[source_type, target_type]
            else:
                conn_prob = conn_between_np[source_type, target_type]

            if np.random.random() < conn_prob:
                adjacency[i, j] = True

    return adjacency
