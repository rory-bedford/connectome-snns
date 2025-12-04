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

                # Calculate total number of edges to create
                total_edges = int(np.round(conn_prob * n_src * n_tgt))

                # We will create more stubs than needed then randomly remove to get exact total_edges count
                expected_src = int(np.ceil(total_edges / n_src))
                expected_tgt = int(np.ceil(total_edges / n_tgt))

                # Shuffle indices once, then tile
                shuffled_src = src_indices.copy()
                np.random.shuffle(shuffled_src)
                shuffled_tgt = tgt_indices.copy()
                np.random.shuffle(shuffled_tgt)

                # Create stub lists by tiling the shuffled indices
                out_stubs = np.tile(shuffled_src, expected_src)[:total_edges]
                in_stubs = np.tile(shuffled_tgt, expected_tgt)[:total_edges]

                # Final shuffle
                np.random.shuffle(out_stubs)
                np.random.shuffle(in_stubs)

                # Clean up repeated edges and self-loops
                while True:
                    # Find repeated edges and self-loops
                    # Create edge pairs
                    edges = np.column_stack([out_stubs, in_stubs])

                    # Find self-loops (where source == target) only if not allowed
                    if not allow_self_loops:
                        self_loop_indices = np.where(out_stubs == in_stubs)[0]
                    else:
                        self_loop_indices = np.array([], dtype=int)

                    # Find repeated edges (excluding first occurrence)
                    # Use edges directly without sorting to detect exact duplicates (i,j) == (i,j)
                    _, first_occurrence_indices, inverse_indices, counts = np.unique(
                        edges,
                        axis=0,
                        return_index=True,
                        return_inverse=True,
                        return_counts=True,
                    )

                    # Vectorized: Find indices where edges are repeated (count > 1) AND it's not the first occurrence
                    # Create mask for first occurrences
                    is_first_occurrence = np.zeros(len(edges), dtype=bool)
                    is_first_occurrence[first_occurrence_indices] = True

                    # Find repeated edges that are not first occurrences
                    is_repeated = counts[inverse_indices] > 1
                    repeated_edge_indices = np.where(
                        is_repeated & ~is_first_occurrence
                    )[0]

                    if len(self_loop_indices) == 0 and len(repeated_edge_indices) == 0:
                        break  # No self-loops or repeated edges, we're done

                    # Combine all problematic indices
                    problematic_indices = np.concatenate(
                        [self_loop_indices, repeated_edge_indices]
                    )

                    # For each problematic index, swap its in_stub with a random position
                    for idx in problematic_indices:
                        # Choose a random position to swap with
                        swap_idx = np.random.randint(0, len(in_stubs))
                        # Swap
                        in_stubs[idx], in_stubs[swap_idx] = (
                            in_stubs[swap_idx],
                            in_stubs[idx],
                        )

                # Match stubs
                adjacency[out_stubs, in_stubs] = True

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
        The "configuration" method uses configuration model for both within-assembly and between-assembly connections.
        For between-assembly connections, each source assembly connects to targets across all other assemblies.
        Assemblies are made equal-sized; any remaining neurons are discarded.
    """
    assert method in ["erdos-renyi", "configuration"], (
        f"method must be 'erdos-renyi' or 'configuration', got '{method}'"
    )

    n_source = len(source_cell_types)
    n_target = len(target_cell_types)

    # Calculate equal assembly sizes
    source_assembly_size = n_source // num_assemblies
    target_assembly_size = n_target // num_assemblies

    # Use only neurons that fit evenly into assemblies
    n_source_used = source_assembly_size * num_assemblies
    n_target_used = target_assembly_size * num_assemblies

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

    # Create equal-sized assembly assignments
    assembly_assignments_source = [
        np.arange(i * source_assembly_size, (i + 1) * source_assembly_size)
        for i in range(num_assemblies)
    ]
    assembly_assignments_target = [
        np.arange(i * target_assembly_size, (i + 1) * target_assembly_size)
        for i in range(num_assemblies)
    ]

    # Create assembly membership arrays (only for neurons in assemblies)
    source_assembly_id = np.repeat(np.arange(num_assemblies), source_assembly_size)
    target_assembly_id = np.repeat(np.arange(num_assemblies), target_assembly_size)

    # Create within-assembly mask using broadcasting
    # Shape: (n_source_used, n_target_used)
    within_assembly_mask = source_assembly_id[:, None] == target_assembly_id

    if method == "erdos-renyi":
        # Get probability matrices for all connections using advanced indexing
        # Only use neurons that are in assemblies
        source_types_used = source_cell_types[:n_source_used]
        target_types_used = target_cell_types[:n_target_used]

        prob_within = conn_within_np[source_types_used[:, None], target_types_used]
        prob_between = conn_between_np[source_types_used[:, None], target_types_used]

        # Select appropriate probabilities based on assembly membership
        prob_matrix = np.where(within_assembly_mask, prob_within, prob_between)

        # Generate random matrix and compare to probabilities
        adjacency = np.random.random((n_source_used, n_target_used)) < prob_matrix

    else:  # method == "configuration"
        adjacency = np.zeros((n_source_used, n_target_used), dtype=bool)

        # Process within-assembly connections using configuration model
        for assembly_id in range(num_assemblies):
            src_indices = assembly_assignments_source[assembly_id]
            tgt_indices = assembly_assignments_target[assembly_id]

            # Get cell types for neurons in this assembly
            src_types = source_cell_types[src_indices]
            tgt_types = target_cell_types[tgt_indices]

            # Use sparse_graph_generator with configuration method for this assembly
            assembly_adjacency = sparse_graph_generator(
                src_types,
                tgt_types,
                conn_within,
                allow_self_loops=allow_self_loops,
                method="configuration",
            )

            # Map assembly adjacency back to global adjacency matrix
            adjacency[np.ix_(src_indices, tgt_indices)] = assembly_adjacency

        # Process between-assembly connections using configuration model
        # Loop over all pairs of source and target assemblies
        for src_assembly_id in range(num_assemblies):
            for tgt_assembly_id in range(num_assemblies):
                # Skip within-assembly connections (already handled above)
                if src_assembly_id == tgt_assembly_id:
                    continue

                src_indices = assembly_assignments_source[src_assembly_id]
                tgt_indices = assembly_assignments_target[tgt_assembly_id]

                # Get cell types for neurons in this source and target assembly pair
                src_types = source_cell_types[src_indices]
                tgt_types = target_cell_types[tgt_indices]

                # Use configuration model for between-assembly connections
                between_adjacency = sparse_graph_generator(
                    src_types,
                    tgt_types,
                    conn_between,
                    allow_self_loops=False,  # Never allow self-loops between assemblies
                    method="configuration",
                )

                # Map between-assembly adjacency back to global adjacency matrix
                adjacency[np.ix_(src_indices, tgt_indices)] = between_adjacency

    # Mask out self-loops if not allowed
    if not allow_self_loops and n_source_used == n_target_used:
        np.fill_diagonal(adjacency, False)

    # Print connection statistics for entire network by cell type
    print("\n" + "=" * 80)
    print("NETWORK CONNECTIVITY STATISTICS (Entire Matrix)")
    print("=" * 80)

    source_types_used = source_cell_types[:n_source_used]
    target_types_used = target_cell_types[:n_target_used]

    # Get unique cell types
    unique_source_types = np.unique(source_types_used)
    unique_target_types = np.unique(target_types_used)

    # Print statistics for each source-target cell type pair across entire network
    for src_type in unique_source_types:
        for tgt_type in unique_target_types:
            src_mask = source_types_used == src_type
            tgt_mask = target_types_used == tgt_type

            # Get submatrix for this cell type pair
            src_indices = np.where(src_mask)[0]
            tgt_indices = np.where(tgt_mask)[0]

            if len(src_indices) == 0 or len(tgt_indices) == 0:
                continue

            submatrix = adjacency[np.ix_(src_indices, tgt_indices)]

            # Incoming connections: for each target, count connections from all sources of src_type
            in_connections_per_target = submatrix.sum(axis=0)
            min_in = in_connections_per_target.min()
            mean_in = in_connections_per_target.mean()
            std_in = in_connections_per_target.std()
            max_in = in_connections_per_target.max()

            # Outgoing connections: for each source, count connections to all targets of tgt_type
            out_connections_per_source = submatrix.sum(axis=1)
            min_out = out_connections_per_source.min()
            mean_out = out_connections_per_source.mean()
            std_out = out_connections_per_source.std()
            max_out = out_connections_per_source.max()

            print(f"  Source Type {src_type} → Target Type {tgt_type}:")
            print(
                f"    In  (to targets):   min={min_in:4d}, mean={mean_in:6.1f}, std={std_in:6.1f}, max={max_in:4d}"
            )
            print(
                f"    Out (from sources): min={min_out:4d}, mean={mean_out:6.1f}, std={std_out:6.1f}, max={max_out:4d}"
            )

    print("\n" + "=" * 80 + "\n")

    return adjacency
