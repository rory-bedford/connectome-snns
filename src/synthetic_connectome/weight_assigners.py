"""Assign continuous weights to graph edges.

This module provides functions to assign continuous weight values to signed
adjacency matrices while preserving Dale's law (excitatory/inhibitory signs).
Separate parameters can be specified for excitatory vs inhibitory connections.

Typical workflow:
    >>> from src.utils.topology_generators import sparse_graph_generator
    >>> connectivity_graph, neuron_types = sparse_graph_generator(n_nodes=100, p=0.1)
    >>> weights = assign_weights_lognormal(connectivity_graph, neuron_types, E_mean=0.0, E_std=1.0,
    ...                                     I_mean=0.5, I_std=0.8)
"""

import numpy as np
from numpy.typing import NDArray

# Type aliases for clarity
IntArray = NDArray[np.int_]
FloatArray = NDArray[np.float64]


def assign_weights_lognormal(
    connectivity_graph: IntArray,
    neuron_types: IntArray,
    E_mean: float = 0.0,
    E_std: float = 1.0,
    I_mean: float = 0.0,
    I_std: float = 1.0,
    seed: int | None = None,
) -> FloatArray:
    """
    Assign log-normal distributed weights to a signed adjacency matrix.

    The magnitude of weights follows a log-normal distribution, with separate
    parameters for excitatory and inhibitory presynaptic neurons. The sign
    is preserved from the input adjacency matrix (Dale's law).

    Args:
        connectivity_graph (IntArray): Signed adjacency matrix with +1/-1 values for edges.
        neuron_types (IntArray): Neuron types array with +1 (excitatory) or -1 (inhibitory).
        E_mean (float): Mean of underlying normal for excitatory connections. Defaults to 0.0.
        E_std (float): Std of underlying normal for excitatory connections. Defaults to 1.0.
        I_mean (float): Mean of underlying normal for inhibitory connections. Defaults to 0.0.
        I_std (float): Std of underlying normal for inhibitory connections. Defaults to 1.0.
        seed (int | None): Random seed for reproducibility. Defaults to None.

    Returns:
        FloatArray: Weighted adjacency matrix with log-normal magnitudes and
            preserved signs.
    """
    rng = np.random.default_rng(seed)
    W = connectivity_graph.astype(np.float64)

    # Find excitatory and inhibitory edges
    E_mask = neuron_types == 1
    I_mask = neuron_types == -1

    # Excitatory edges: where presynaptic neuron is excitatory
    E_edges = (connectivity_graph != 0) & E_mask[:, np.newaxis]
    n_E_edges = E_edges.sum()

    # Inhibitory edges: where presynaptic neuron is inhibitory
    I_edges = (connectivity_graph != 0) & I_mask[:, np.newaxis]
    n_I_edges = I_edges.sum()

    # Generate weights for excitatory connections
    if n_E_edges > 0:
        E_weights = rng.lognormal(E_mean, E_std, size=n_E_edges)
        W[E_edges] = E_weights

    # Generate weights for inhibitory connections (negative)
    if n_I_edges > 0:
        I_weights = rng.lognormal(I_mean, I_std, size=n_I_edges)
        W[I_edges] = -I_weights

    return W


def assign_weights_gamma(
    connectivity_graph: IntArray,
    neuron_types: IntArray,
    E_shape: float = 2.0,
    E_scale: float = 1.0,
    I_shape: float = 2.0,
    I_scale: float = 1.0,
    seed: int | None = None,
) -> FloatArray:
    """
    Assign gamma-distributed weights to a signed adjacency matrix.

    The magnitude of weights follows a gamma distribution, with separate
    parameters for excitatory and inhibitory presynaptic neurons. The sign
    is preserved from the input adjacency matrix (Dale's law).

    Args:
        connectivity_graph (IntArray): Signed adjacency matrix with +1/-1 values for edges.
        neuron_types (IntArray): Neuron types array with +1 (excitatory) or -1 (inhibitory).
        E_shape (float): Shape parameter (k) for excitatory connections. Defaults to 2.0.
        E_scale (float): Scale parameter (θ) for excitatory connections. Defaults to 1.0.
        I_shape (float): Shape parameter (k) for inhibitory connections. Defaults to 2.0.
        I_scale (float): Scale parameter (θ) for inhibitory connections. Defaults to 1.0.
        seed (int | None): Random seed for reproducibility. Defaults to None.

    Returns:
        FloatArray: Weighted adjacency matrix with gamma-distributed magnitudes
            and preserved signs.
    """
    rng = np.random.default_rng(seed)
    W = connectivity_graph.astype(np.float64)

    # Find excitatory and inhibitory edges
    E_mask = neuron_types == 1
    I_mask = neuron_types == -1

    # Excitatory edges: where presynaptic neuron is excitatory
    E_edges = (connectivity_graph != 0) & E_mask[:, np.newaxis]
    n_E_edges = E_edges.sum()

    # Inhibitory edges: where presynaptic neuron is inhibitory
    I_edges = (connectivity_graph != 0) & I_mask[:, np.newaxis]
    n_I_edges = I_edges.sum()

    # Generate weights for excitatory connections
    if n_E_edges > 0:
        E_weights = rng.gamma(E_shape, E_scale, size=n_E_edges)
        W[E_edges] = E_weights

    # Generate weights for inhibitory connections (negative)
    if n_I_edges > 0:
        I_weights = rng.gamma(I_shape, I_scale, size=n_I_edges)
        W[I_edges] = -I_weights

    return W


def assign_weights_uniform(
    connectivity_graph: IntArray,
    neuron_types: IntArray,
    E_low: float = 0.1,
    E_high: float = 1.0,
    I_low: float = 0.1,
    I_high: float = 1.0,
    seed: int | None = None,
) -> FloatArray:
    """
    Assign uniformly distributed weights to a signed adjacency matrix.

    The magnitude of weights follows a uniform distribution, with separate
    parameters for excitatory and inhibitory presynaptic neurons. The sign
    is preserved from the input adjacency matrix (Dale's law). Useful for
    baseline comparisons.

    Args:
        connectivity_graph (IntArray): Signed adjacency matrix with +1/-1 values for edges.
        neuron_types (IntArray): Neuron types array with +1 (excitatory) or -1 (inhibitory).
        E_low (float): Lower bound for excitatory connections. Defaults to 0.1.
        E_high (float): Upper bound for excitatory connections. Defaults to 1.0.
        I_low (float): Lower bound for inhibitory connections. Defaults to 0.1.
        I_high (float): Upper bound for inhibitory connections. Defaults to 1.0.
        seed (int | None): Random seed for reproducibility. Defaults to None.

    Returns:
        FloatArray: Weighted adjacency matrix with uniformly distributed
            magnitudes and preserved signs.
    """
    rng = np.random.default_rng(seed)
    W = connectivity_graph.astype(np.float64)

    # Find excitatory and inhibitory edges
    E_mask = neuron_types == 1
    I_mask = neuron_types == -1

    # Excitatory edges: where presynaptic neuron is excitatory
    E_edges = (connectivity_graph != 0) & E_mask[:, np.newaxis]
    n_E_edges = E_edges.sum()

    # Inhibitory edges: where presynaptic neuron is inhibitory
    I_edges = (connectivity_graph != 0) & I_mask[:, np.newaxis]
    n_I_edges = I_edges.sum()

    # Generate weights for excitatory connections
    if n_E_edges > 0:
        E_weights = rng.uniform(E_low, E_high, size=n_E_edges)
        W[E_edges] = E_weights

    # Generate weights for inhibitory connections (negative)
    if n_I_edges > 0:
        I_weights = rng.uniform(I_low, I_high, size=n_I_edges)
        W[I_edges] = -I_weights

    return W
