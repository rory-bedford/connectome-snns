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
    method: str = "erdos-renyi",
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
        method (str): Method for graph generation. Either "erdos-renyi" or "configuration".

    Returns:
        BoolArray: Boolean adjacency matrix of shape (len(source_cell_types), len(target_cell_types)).

    Note:
        For recurrent connections, pass the same array for both source_cell_types and target_cell_types.
        The "configuration" method generates graphs with fixed degree sequences sampled from the
        connectivity probabilities, while "erdos-renyi" samples each edge independently.
    """
    assert method in ["erdos-renyi", "configuration"], (
        f"method must be 'erdos-renyi' or 'configuration', got '{method}'"
    )

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

    if method == "erdos-renyi":
        # Use advanced indexing to get probability matrix for all connections
        prob_matrix = conn_matrix_np[source_cell_types[:, None], target_cell_types]

        # Generate random matrix and compare to probabilities
        adjacency = np.random.random((n_source, n_target)) < prob_matrix

    else:  # method == "configuration"
        # Configuration model: process each cell type pair separately
        adjacency = np.zeros((n_source, n_target), dtype=bool)

        # Loop over all combinations of source and target cell types
        for src_type in range(n_source_types):
            for tgt_type in range(n_target_types):
                # Get connection probability for this cell type pair
                conn_prob = conn_matrix_np[src_type, tgt_type]

                if conn_prob == 0:
                    continue

                # Find all neurons of these types
                src_mask = source_cell_types == src_type
                tgt_mask = target_cell_types == tgt_type
                src_indices = np.where(src_mask)[0]
                tgt_indices = np.where(tgt_mask)[0]

                n_src = len(src_indices)
                n_tgt = len(tgt_indices)

                if n_src == 0 or n_tgt == 0:
                    continue

                # Each source neuron gets conn_prob * n_tgt connections
                # Each target neuron gets conn_prob * n_src connections
                expected_out_per_source = conn_prob * n_tgt
                expected_in_per_target = conn_prob * n_src

                out_degrees = np.round(np.full(n_src, expected_out_per_source)).astype(
                    int
                )
                in_degrees = np.round(np.full(n_tgt, expected_in_per_target)).astype(
                    int
                )

                # Balance stubs for this cell type pair (may not be perfectly balanced due to rounding)
                total_out = out_degrees.sum()
                total_in = in_degrees.sum()

                if total_out > total_in:
                    deficit = total_out - total_in
                    indices = np.random.choice(
                        n_tgt, min(deficit, n_tgt), replace=False
                    )
                    in_degrees[indices] += 1
                elif total_in > total_out:
                    deficit = total_in - total_out
                    indices = np.random.choice(
                        n_src, min(deficit, n_src), replace=False
                    )
                    out_degrees[indices] += 1

                # Create stub lists for this cell type pair
                out_stubs = np.repeat(src_indices, out_degrees)
                in_stubs = np.repeat(tgt_indices, in_degrees)

                # Shuffle stubs
                np.random.shuffle(out_stubs)
                np.random.shuffle(in_stubs)

                # Match stubs - lengths are equal after balancing
                adjacency[out_stubs, in_stubs] = True

    # Mask out self-loops if not allowed
    if not allow_self_loops and n_source == n_target:
        np.fill_diagonal(adjacency, False)

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
        source_cell_types,
        target_cell_types,
        conn_matrix,
        allow_self_loops,
        method="erdos-renyi",
    )


def assembly_generator(
    source_cell_types: IntArray,
    target_cell_types: IntArray,
    num_assemblies: int,
    conn_within: List[List[float]],
    conn_between: List[List[float]],
    allow_self_loops: bool = False,
    method: str = "erdos-renyi",
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
        method (str): Method for graph generation. Either "erdos-renyi" or "configuration".

    Returns:
        BoolArray: Boolean adjacency matrix of shape (len(source_cell_types), len(target_cell_types)).

    Note:
        For recurrent connections, pass the same array for both source_cell_types and target_cell_types.
        The "configuration" method uses configuration model within assemblies and Erdős-Rényi between assemblies.
    """
    assert method in ["erdos-renyi", "configuration"], (
        f"method must be 'erdos-renyi' or 'configuration', got '{method}'"
    )

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

    # Create assembly membership arrays
    source_assembly_id = np.zeros(n_source, dtype=int)
    target_assembly_id = np.zeros(n_target, dtype=int)

    for assembly_id, (source_indices, target_indices) in enumerate(
        zip(assembly_assignments_source, assembly_assignments_target)
    ):
        source_assembly_id[source_indices] = assembly_id
        target_assembly_id[target_indices] = assembly_id

    # Create within-assembly mask using broadcasting
    # Shape: (n_source, n_target)
    within_assembly_mask = source_assembly_id[:, None] == target_assembly_id

    if method == "erdos-renyi":
        # Get probability matrices for all connections using advanced indexing
        prob_within = conn_within_np[source_cell_types[:, None], target_cell_types]
        prob_between = conn_between_np[source_cell_types[:, None], target_cell_types]

        # Select appropriate probabilities based on assembly membership
        prob_matrix = np.where(within_assembly_mask, prob_within, prob_between)

        # Generate random matrix and compare to probabilities
        adjacency = np.random.random((n_source, n_target)) < prob_matrix

    else:  # method == "configuration"
        adjacency = np.zeros((n_source, n_target), dtype=bool)

        # Process within-assembly connections using configuration model
        for assembly_id in range(num_assemblies):
            src_indices = assembly_assignments_source[assembly_id]
            tgt_indices = assembly_assignments_target[assembly_id]

            # Get cell types for this assembly
            assembly_src_types = source_cell_types[src_indices]
            assembly_tgt_types = target_cell_types[tgt_indices]

            # Use sparse_graph_generator with configuration method for this assembly
            assembly_adjacency = sparse_graph_generator(
                assembly_src_types,
                assembly_tgt_types,
                conn_within,
                allow_self_loops=allow_self_loops,
                method="configuration",
            )

            # Map assembly adjacency back to global adjacency matrix
            adjacency[np.ix_(src_indices, tgt_indices)] = assembly_adjacency

        # Process between-assembly connections using Erdős-Rényi
        prob_between = conn_between_np[source_cell_types[:, None], target_cell_types]
        between_assembly_mask = ~within_assembly_mask

        # Generate random connections for between-assembly pairs
        random_matrix = np.random.random((n_source, n_target))
        between_connections = (random_matrix < prob_between) & between_assembly_mask
        adjacency = adjacency | between_connections

    # Mask out self-loops if not allowed
    if not allow_self_loops and n_source == n_target:
        np.fill_diagonal(adjacency, False)

    return adjacency
