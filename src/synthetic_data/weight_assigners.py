"""Assign continuous weights to graph edges.

This module provides functions to assign continuous weight values to signed
adjacency matrices while preserving Dale's law (excitatory/inhibitory signs).

Typical workflow:
    >>> from src.utils.topology_generators import sparse_graph_generator
    >>> adj, types = sparse_graph_generator(n_nodes=100, p=0.1)
    >>> weights = assign_weights_lognormal(adj, mean=0.0, std=1.0)
"""

import numpy as np
from numpy.typing import NDArray


def assign_weights_lognormal(
    adj: NDArray[np.int_],
    mean: float = 0.0,
    std: float = 1.0,
    seed: int | None = None,
) -> NDArray[np.float64]:
    """
    Assign log-normal distributed weights to a signed adjacency matrix.

    The magnitude of weights follows a log-normal distribution, while the sign
    is preserved from the input adjacency matrix (Dale's law).

    Args:
        adj (NDArray[np.int_]): Signed adjacency matrix with +1/-1 values for edges.
        mean (float): Mean of the underlying normal distribution. Defaults to 0.0.
        std (float): Standard deviation of the underlying normal distribution. Defaults to 1.0.
        seed (int | None): Random seed for reproducibility. Defaults to None.

    Returns:
        NDArray[np.float64]: Weighted adjacency matrix with log-normal magnitudes and
            preserved signs.
    """
    rng = np.random.default_rng(seed)
    W = adj.astype(np.float64)

    # Find edges (non-zero entries)
    edges = W != 0
    n_edges = edges.sum()

    # Generate positive weights from log-normal distribution
    weights = rng.lognormal(mean, std, size=n_edges)

    # Assign weights preserving sign
    signs = np.sign(W[edges])
    W[edges] = signs * weights

    return W


def assign_weights_gamma(
    adj: NDArray[np.int_],
    shape: float = 2.0,
    scale: float = 1.0,
    seed: int | None = None,
) -> NDArray[np.float64]:
    """
    Assign gamma-distributed weights to a signed adjacency matrix.

    The magnitude of weights follows a gamma distribution, while the sign
    is preserved from the input adjacency matrix (Dale's law).

    Args:
        adj (NDArray[np.int_]): Signed adjacency matrix with +1/-1 values for edges.
        shape (float): Shape parameter (k) of the gamma distribution. Defaults to 2.0.
        scale (float): Scale parameter (θ) of the gamma distribution. Defaults to 1.0.
        seed (int | None): Random seed for reproducibility. Defaults to None.

    Returns:
        NDArray[np.float64]: Weighted adjacency matrix with gamma-distributed magnitudes
            and preserved signs.
    """
    rng = np.random.default_rng(seed)
    W = adj.astype(np.float64)

    # Find edges (non-zero entries)
    edges = W != 0
    n_edges = edges.sum()

    # Generate positive weights from gamma distribution
    weights = rng.gamma(shape, scale, size=n_edges)

    # Assign weights preserving sign
    signs = np.sign(W[edges])
    W[edges] = signs * weights

    return W


def assign_weights_uniform(
    adj: NDArray[np.int_],
    low: float = 0.1,
    high: float = 1.0,
    seed: int | None = None,
) -> NDArray[np.float64]:
    """
    Assign uniformly distributed weights to a signed adjacency matrix.

    The magnitude of weights follows a uniform distribution, while the sign
    is preserved from the input adjacency matrix (Dale's law). Useful for
    baseline comparisons.

    Args:
        adj (NDArray[np.int_]): Signed adjacency matrix with +1/-1 values for edges.
        low (float): Lower bound of the uniform distribution. Defaults to 0.1.
        high (float): Upper bound of the uniform distribution. Defaults to 1.0.
        seed (int | None): Random seed for reproducibility. Defaults to None.

    Returns:
        NDArray[np.float64]: Weighted adjacency matrix with uniformly distributed
            magnitudes and preserved signs.
    """
    rng = np.random.default_rng(seed)
    W = adj.astype(np.float64)

    # Find edges (non-zero entries)
    edges = W != 0
    n_edges = edges.sum()

    # Generate positive weights from uniform distribution
    weights = rng.uniform(low, high, size=n_edges)

    # Assign weights preserving sign
    signs = np.sign(W[edges])
    W[edges] = signs * weights

    return W
