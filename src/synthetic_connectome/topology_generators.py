"""Generate graph topologies with neuron type assignments.

This module provides functions to generate binary adjacency matrices with
excitatory (+1) and inhibitory (-1) neuron types following Dale's law.

Typical workflow:
    >>> adj, types = sparse_graph_generator(n_nodes=100, p=0.1, p_exc=0.8)
    >>> # adj is a signed adjacency matrix with +1/-1 values
    >>> # types is an array of neuron types (+1 or -1)

    >>> # For feedforward connections:
    >>> adj, types = sparse_graph_generator(n_nodes=(100, 50), p=0.1, p_exc=0.8)
    >>> # adj is a (100, 50) matrix, types has shape (100,)
"""

import numpy as np
from numpy.typing import NDArray

# Type aliases for clarity
IntArray = NDArray[np.int_]
BoolArray = NDArray[np.bool_]


def sparse_graph_generator(
    n_nodes: int | tuple[int, int],
    p: float,
    p_exc: float,
    seed: int | None = None,
) -> tuple[IntArray, IntArray]:
    """
    Generate a sparse random graph using the Erdos-Renyi model with neuron types.

    In the Erdos-Renyi model, each possible edge exists independently with probability p.
    Neurons are assigned as excitatory (+1) or inhibitory (-1) following Dale's law.

    Args:
        n_nodes (int | tuple[int, int]): Number of nodes in the graph. If int, creates
            a square (n_nodes, n_nodes) matrix. If tuple (n_source, n_target), creates
            a rectangular (n_source, n_target) matrix for feedforward connections.
        p (float): Probability of edge creation between any two nodes.
        p_exc (float): Proportion of excitatory neurons.
        seed (int | None): Random seed for reproducibility. Defaults to None.

    Returns:
        tuple[IntArray, IntArray]: A tuple containing:
            - Signed adjacency matrix of shape (n_source, n_target) with +1/-1 values.
            - Neuron types array of shape (n_source,) with +1 (excitatory) or -1 (inhibitory).
    """
    rng = np.random.default_rng(seed)

    # Parse n_nodes into source and target dimensions
    if isinstance(n_nodes, int):
        n_source = n_target = n_nodes
    else:
        n_source, n_target = n_nodes

    # Generate binary adjacency matrix
    adjacency = rng.random((n_source, n_target)) < p

    # No self-loops only for square matrices
    if n_source == n_target:
        np.fill_diagonal(adjacency, 0)

    # Assign neuron types to source neurons
    neuron_types = np.ones(n_source, dtype=np.int_)
    n_inh = int(n_source * (1 - p_exc))
    inh_idx = rng.choice(n_source, size=n_inh, replace=False)
    neuron_types[inh_idx] = -1

    # Apply Dale's law: edges take sign of presynaptic neuron
    signed_adjacency = adjacency.astype(np.int_) * neuron_types[:, np.newaxis]

    return signed_adjacency, neuron_types


def dense_graph_generator(
    n_nodes: int | tuple[int, int],
    p_exc: float,
    seed: int | None = None,
) -> tuple[IntArray, IntArray]:
    """
    Generate a fully connected (dense) graph with neuron types.

    All possible edges exist except self-loops (in square matrices). Neurons are assigned as
    excitatory (+1) or inhibitory (-1) following Dale's law.

    Args:
        n_nodes (int | tuple[int, int]): Number of nodes in the graph. If int, creates
            a square (n_nodes, n_nodes) matrix. If tuple (n_source, n_target), creates
            a rectangular (n_source, n_target) matrix for feedforward connections.
        p_exc (float): Proportion of excitatory neurons.
        seed (int | None): Random seed for reproducibility. Defaults to None.

    Returns:
        tuple[IntArray, IntArray]: A tuple containing:
            - Signed adjacency matrix of shape (n_source, n_target) with +1/-1 values.
            - Neuron types array of shape (n_source,) with +1 (excitatory) or -1 (inhibitory).
    """
    rng = np.random.default_rng(seed)

    # Parse n_nodes into source and target dimensions
    if isinstance(n_nodes, int):
        n_source = n_target = n_nodes
    else:
        n_source, n_target = n_nodes

    # Generate fully connected adjacency matrix
    adjacency = np.ones((n_source, n_target), dtype=np.bool_)

    # No self-loops only for square matrices
    if n_source == n_target:
        np.fill_diagonal(adjacency, 0)

    # Assign neuron types to source neurons
    neuron_types = np.ones(n_source, dtype=np.int_)
    n_inh = int(n_source * (1 - p_exc))
    inh_idx = rng.choice(n_source, size=n_inh, replace=False)
    neuron_types[inh_idx] = -1

    # Apply Dale's law: edges take sign of presynaptic neuron
    signed_adjacency = adjacency.astype(np.int_) * neuron_types[:, np.newaxis]

    return signed_adjacency, neuron_types


def assembly_generator(
    n_nodes: int | tuple[int, int],
    n_assemblies: int,
    p_within: float,
    p_between: float,
    p_exc: float,
    seed: int | None = None,
) -> tuple[IntArray, IntArray]:
    """
    Generate a graph with assembly structure and neuron types.

    Nodes are assigned to a predetermined number of assemblies. Connections within
    assemblies occur with probability p_within, and connections between assemblies
    occur with probability p_between. Neurons are assigned as excitatory (+1) or
    inhibitory (-1) following Dale's law.

    Note: For rectangular matrices (feedforward), assemblies are only defined for
    source neurons. Target neurons do not have assembly structure.

    Args:
        n_nodes (int | tuple[int, int]): Number of nodes in the graph. If int, creates
            a square (n_nodes, n_nodes) matrix. If tuple (n_source, n_target), creates
            a rectangular (n_source, n_target) matrix for feedforward connections.
        n_assemblies (int): Number of assemblies to create (only applies to source neurons).
        p_within (float): Probability of connection within the same assembly.
        p_between (float): Probability of connection between different assemblies.
        p_exc (float): Proportion of excitatory neurons.
        seed (int | None): Random seed for reproducibility. Defaults to None.

    Returns:
        tuple[IntArray, IntArray]: A tuple containing:
            - Signed adjacency matrix of shape (n_source, n_target) with +1/-1 values.
            - Neuron types array of shape (n_source,) with +1 (excitatory) or -1 (inhibitory).
    """
    # Parse n_nodes into source and target dimensions
    if isinstance(n_nodes, int):
        n_source = n_target = n_nodes
    else:
        n_source, n_target = n_nodes

    if n_assemblies > n_source:
        raise ValueError("Number of assemblies cannot exceed number of source nodes.")

    rng = np.random.default_rng(seed)

    # Assign source nodes to assemblies (roughly equal sizes)
    assembly_assignments = np.array_split(np.arange(n_source), n_assemblies)

    # Initialize adjacency matrix
    adjacency = np.zeros((n_source, n_target), dtype=np.bool_)

    # Generate connections
    for i in range(n_source):
        for j in range(n_target):
            # Skip self-loops for square matrices
            if n_source == n_target and i == j:
                continue

            # Determine which assembly node i belongs to
            assembly_i = next(
                k for k, assembly in enumerate(assembly_assignments) if i in assembly
            )

            # For rectangular matrices, determine assembly of target node j
            if n_source == n_target:
                assembly_j = next(
                    k
                    for k, assembly in enumerate(assembly_assignments)
                    if j in assembly
                )
            else:
                # For feedforward, assign target nodes to assemblies based on index
                assembly_j = j * n_assemblies // n_target

            # Choose probability based on whether nodes are in same assembly
            p = p_within if assembly_i == assembly_j else p_between

            # Create edge with appropriate probability
            if rng.random() < p:
                adjacency[i, j] = True

    # Assign neuron types to source neurons
    neuron_types = np.ones(n_source, dtype=np.int_)
    n_inh = int(n_source * (1 - p_exc))
    inh_idx = rng.choice(n_source, size=n_inh, replace=False)
    neuron_types[inh_idx] = -1

    # Apply Dale's law: edges take sign of presynaptic neuron
    signed_adjacency = adjacency.astype(np.int_) * neuron_types[:, np.newaxis]

    return signed_adjacency, neuron_types
