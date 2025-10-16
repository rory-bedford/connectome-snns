"""Generate graph topologies with neuron type assignments.

This module provides functions to generate binary adjacency matrices with
excitatory (+1) and inhibitory (-1) neuron types following Dale's law.

Typical workflow:
    >>> adj, types = sparse_graph_generator(num_neurons=100, p=0.1, p_E=0.8)
    >>> # adj is a signed adjacency matrix with +1/-1 values
    >>> # types is an array of neuron types (+1 or -1)

    >>> # For feedforward connections:
    >>> adj, types = sparse_graph_generator(num_neurons=(100, 50), p=0.1, p_E=0.8)
    >>> # adj is a (100, 50) matrix, types has shape (100,)
"""

import numpy as np
from numpy.typing import NDArray

# Type aliases for clarity
IntArray = NDArray[np.int_]
BoolArray = NDArray[np.bool_]


def sparse_graph_generator(
    num_neurons: int | tuple[int, int],
    p: float,
    p_E: float,
    seed: int | None = None,
) -> tuple[IntArray, IntArray]:
    """
    Generate a sparse random graph using the Erdos-Renyi model with neuron types.

    In the Erdos-Renyi model, each possible edge exists independently with probability p.
    Neurons are assigned as excitatory (+1) or inhibitory (-1) following Dale's law.

    Args:
        num_neurons (int | tuple[int, int]): Number of nodes in the graph. If int, creates
            a square (num_neurons, num_neurons) matrix. If tuple (n_source, n_target), creates
            a rectangular (n_source, n_target) matrix for feedforward connections.
        p (float): Probability of edge creation between any two nodes.
        p_E (float): Proportion of excitatory neurons.
        seed (int | None): Random seed for reproducibility. Defaults to None.

    Returns:
        tuple[IntArray, IntArray]: A tuple containing:
            - Signed adjacency matrix of shape (n_source, n_target) with +1/-1 values.
            - Neuron types array of shape (n_source,) with +1 (excitatory) or -1 (inhibitory).
    """
    rng = np.random.default_rng(seed)

    # Parse num_neurons into source and target dimensions
    if isinstance(num_neurons, int):
        n_source = n_target = num_neurons
    else:
        n_source, n_target = num_neurons

    # Generate binary adjacency matrix
    adjacency = rng.random((n_source, n_target)) < p

    # No self-loops only for square matrices
    if n_source == n_target:
        np.fill_diagonal(adjacency, 0)

    # Assign neuron types to source neurons
    neuron_types = np.ones(n_source, dtype=np.int_)
    n_inh = int(n_source * (1 - p_E))
    inh_idx = rng.choice(n_source, size=n_inh, replace=False)
    neuron_types[inh_idx] = -1

    # Apply Dale's law: edges take sign of presynaptic neuron
    signed_adjacency = adjacency.astype(np.int_) * neuron_types[:, np.newaxis]

    return signed_adjacency, neuron_types


def dense_graph_generator(
    num_neurons: int | tuple[int, int],
    p_E: float,
    seed: int | None = None,
) -> tuple[IntArray, IntArray]:
    """
    Generate a fully connected (dense) graph with neuron types.

    All possible edges exist except self-loops (in square matrices). Neurons are assigned as
    excitatory (+1) or inhibitory (-1) following Dale's law.

    Args:
        num_neurons (int | tuple[int, int]): Number of nodes in the graph. If int, creates
            a square (num_neurons, num_neurons) matrix. If tuple (n_source, n_target), creates
            a rectangular (n_source, n_target) matrix for feedforward connections.
        p_E (float): Proportion of excitatory neurons.
        seed (int | None): Random seed for reproducibility. Defaults to None.

    Returns:
        tuple[IntArray, IntArray]: A tuple containing:
            - Signed adjacency matrix of shape (n_source, n_target) with +1/-1 values.
            - Neuron types array of shape (n_source,) with +1 (excitatory) or -1 (inhibitory).
    """
    rng = np.random.default_rng(seed)

    # Parse num_neurons into source and target dimensions
    if isinstance(num_neurons, int):
        n_source = n_target = num_neurons
    else:
        n_source, n_target = num_neurons

    # Generate fully connected adjacency matrix
    adjacency = np.ones((n_source, n_target), dtype=np.bool_)

    # No self-loops only for square matrices
    if n_source == n_target:
        np.fill_diagonal(adjacency, 0)

    # Assign neuron types to source neurons
    neuron_types = np.ones(n_source, dtype=np.int_)
    n_inh = int(n_source * (1 - p_E))
    inh_idx = rng.choice(n_source, size=n_inh, replace=False)
    neuron_types[inh_idx] = -1

    # Apply Dale's law: edges take sign of presynaptic neuron
    signed_adjacency = adjacency.astype(np.int_) * neuron_types[:, np.newaxis]

    return signed_adjacency, neuron_types


def assembly_generator(
    num_neurons: int | tuple[int, int],
    num_assemblies: int,
    p_within: float,
    p_between: float,
    p_E: float,
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
        num_neurons (int | tuple[int, int]): Number of nodes in the graph. If int, creates
            a square (num_neurons, num_neurons) matrix. If tuple (n_source, n_target), creates
            a rectangular (n_source, n_target) matrix for feedforward connections.
        num_assemblies (int): Number of assemblies to create (only applies to source neurons).
        p_within (float): Probability of connection within the same assembly.
        p_between (float): Probability of connection between different assemblies.
        p_E (float): Proportion of excitatory neurons.
        seed (int | None): Random seed for reproducibility. Defaults to None.

    Returns:
        tuple[IntArray, IntArray]: A tuple containing:
            - Signed adjacency matrix of shape (n_source, n_target) with +1/-1 values.
            - Neuron types array of shape (n_source,) with +1 (excitatory) or -1 (inhibitory).
    """
    # Parse num_neurons into source and target dimensions
    if isinstance(num_neurons, int):
        n_source = n_target = num_neurons
    else:
        n_source, n_target = num_neurons

    if num_assemblies > n_source:
        raise ValueError("Number of assemblies cannot exceed number of source nodes.")

    rng = np.random.default_rng(seed)

    # Assign source and target nodes to assemblies (roughly equal sizes)
    assembly_assignments_source = np.array_split(np.arange(n_source), num_assemblies)
    assembly_assignments_target = np.array_split(np.arange(n_target), num_assemblies)

    # Create within-assembly mask using outer products
    within_assembly_mask = np.zeros((n_source, n_target), dtype=np.bool_)
    for source_assembly, target_assembly in zip(
        assembly_assignments_source, assembly_assignments_target
    ):
        # Create indicator vectors for this assembly
        source_in_assembly = np.isin(np.arange(n_source), source_assembly)
        target_in_assembly = np.isin(np.arange(n_target), target_assembly)

        # Outer product gives True where both source and target are in this assembly
        within_assembly_mask |= np.outer(source_in_assembly, target_in_assembly)

    # Generate random connections based on within/between assembly probabilities
    rand_matrix = rng.random((n_source, n_target))
    adjacency = np.where(
        within_assembly_mask,
        rand_matrix < p_within,
        rand_matrix < p_between,
    )

    # No self-loops for square matrices
    if n_source == n_target:
        np.fill_diagonal(adjacency, 0)

    # Assign neuron types to source neurons
    neuron_types = np.ones(n_source, dtype=np.int_)
    n_inh = int(n_source * (1 - p_E))
    inh_idx = rng.choice(n_source, size=n_inh, replace=False)
    neuron_types[inh_idx] = -1

    # Apply Dale's law: edges take sign of presynaptic neuron
    signed_adjacency = adjacency.astype(np.int_) * neuron_types[:, np.newaxis]

    return signed_adjacency, neuron_types
