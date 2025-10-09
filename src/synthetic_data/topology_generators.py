"""Generate graph topologies with neuron type assignments.

This module provides functions to generate binary adjacency matrices with
excitatory (+1) and inhibitory (-1) neuron types following Dale's law.

Typical workflow:
    >>> adj, types = sparse_graph_generator(n_nodes=100, p=0.1, p_exc=0.8)
    >>> # adj is a signed adjacency matrix with +1/-1 values
    >>> # types is an array of neuron types (+1 or -1)
"""

import numpy as np
from numpy.typing import NDArray

# Type aliases for clarity
IntArray = NDArray[np.int_]
BoolArray = NDArray[np.bool_]


def sparse_graph_generator(
    n_nodes: int,
    p: float,
    p_exc: float = 0.8,
    seed: int | None = None,
) -> tuple[IntArray, IntArray]:
    """
    Generate a sparse random graph using the Erdos-Renyi model with neuron types.

    In the Erdos-Renyi model, each possible edge exists independently with probability p.
    Neurons are assigned as excitatory (+1) or inhibitory (-1) following Dale's law.

    Args:
        n_nodes (int): Number of nodes in the graph.
        p (float): Probability of edge creation between any two nodes.
        p_exc (float): Proportion of excitatory neurons. Defaults to 0.8 (cortical ratio).
        seed (int | None): Random seed for reproducibility. Defaults to None.

    Returns:
        tuple[IntArray, IntArray]: A tuple containing:
            - Signed adjacency matrix of shape (n_nodes, n_nodes) with +1/-1 values.
            - Neuron types array of shape (n_nodes,) with +1 (excitatory) or -1 (inhibitory).
    """
    rng = np.random.default_rng(seed)

    # Generate binary adjacency matrix
    adjacency = rng.random((n_nodes, n_nodes)) < p
    np.fill_diagonal(adjacency, 0)  # No self-loops

    # Assign neuron types
    neuron_types = np.ones(n_nodes, dtype=np.int_)
    n_inh = int(n_nodes * (1 - p_exc))
    inh_idx = rng.choice(n_nodes, size=n_inh, replace=False)
    neuron_types[inh_idx] = -1

    # Apply Dale's law: edges take sign of presynaptic neuron
    signed_adjacency = adjacency.astype(np.int_) * neuron_types[:, np.newaxis]

    return signed_adjacency, neuron_types


def dense_graph_generator(
    n_nodes: int,
    p_exc: float = 0.8,
    seed: int | None = None,
) -> tuple[IntArray, IntArray]:
    """
    Generate a fully connected (dense) graph with neuron types.

    All possible edges exist except self-loops. Neurons are assigned as
    excitatory (+1) or inhibitory (-1) following Dale's law.

    Args:
        n_nodes (int): Number of nodes in the graph.
        p_exc (float): Proportion of excitatory neurons. Defaults to 0.8 (cortical ratio).
        seed (int | None): Random seed for reproducibility. Defaults to None.

    Returns:
        tuple[IntArray, IntArray]: A tuple containing:
            - Signed adjacency matrix of shape (n_nodes, n_nodes) with +1/-1 values.
            - Neuron types array of shape (n_nodes,) with +1 (excitatory) or -1 (inhibitory).
    """
    rng = np.random.default_rng(seed)

    # Generate fully connected adjacency matrix
    adjacency = np.ones((n_nodes, n_nodes), dtype=np.bool_)
    np.fill_diagonal(adjacency, 0)  # No self-loops

    # Assign neuron types
    neuron_types = np.ones(n_nodes, dtype=np.int_)
    n_inh = int(n_nodes * (1 - p_exc))
    inh_idx = rng.choice(n_nodes, size=n_inh, replace=False)
    neuron_types[inh_idx] = -1

    # Apply Dale's law: edges take sign of presynaptic neuron
    signed_adjacency = adjacency.astype(np.int_) * neuron_types[:, np.newaxis]

    return signed_adjacency, neuron_types


def assembly_generator(
    n_nodes: int,
    n_assemblies: int,
    p_within: float,
    p_between: float,
    p_exc: float = 0.8,
    seed: int | None = None,
) -> tuple[IntArray, IntArray]:
    """
    Generate a graph with assembly structure and neuron types.

    Nodes are assigned to a predetermined number of assemblies. Connections within
    assemblies occur with probability p_within, and connections between assemblies
    occur with probability p_between. Neurons are assigned as excitatory (+1) or
    inhibitory (-1) following Dale's law.

    Args:
        n_nodes (int): Number of nodes in the graph.
        n_assemblies (int): Number of assemblies to create.
        p_within (float): Probability of connection within the same assembly.
        p_between (float): Probability of connection between different assemblies.
        p_exc (float): Proportion of excitatory neurons. Defaults to 0.8 (cortical ratio).
        seed (int | None): Random seed for reproducibility. Defaults to None.

    Returns:
        tuple[IntArray, IntArray]: A tuple containing:
            - Signed adjacency matrix of shape (n_nodes, n_nodes) with +1/-1 values.
            - Neuron types array of shape (n_nodes,) with +1 (excitatory) or -1 (inhibitory).
    """
    if n_assemblies > n_nodes:
        raise ValueError("Number of assemblies cannot exceed number of nodes.")

    rng = np.random.default_rng(seed)

    # Assign nodes to assemblies (roughly equal sizes)
    assembly_assignments = np.array_split(np.arange(n_nodes), n_assemblies)

    # Initialize adjacency matrix
    adjacency = np.zeros((n_nodes, n_nodes), dtype=np.bool_)

    # Generate connections
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i == j:
                continue  # No self-loops

            # Determine which assemblies nodes i and j belong to
            assembly_i = next(
                k for k, assembly in enumerate(assembly_assignments) if i in assembly
            )
            assembly_j = next(
                k for k, assembly in enumerate(assembly_assignments) if j in assembly
            )

            # Choose probability based on whether nodes are in same assembly
            p = p_within if assembly_i == assembly_j else p_between

            # Create edge with appropriate probability
            if rng.random() < p:
                adjacency[i, j] = True

    # Assign neuron types
    neuron_types = np.ones(n_nodes, dtype=np.int_)
    n_inh = int(n_nodes * (1 - p_exc))
    inh_idx = rng.choice(n_nodes, size=n_inh, replace=False)
    neuron_types[inh_idx] = -1

    # Apply Dale's law: edges take sign of presynaptic neuron
    signed_adjacency = adjacency.astype(np.int_) * neuron_types[:, np.newaxis]

    return signed_adjacency, neuron_types
