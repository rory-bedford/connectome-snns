"""Assign continuous weights to graph edges.

This module provides functions to assign continuous weight values to boolean
adjacency matrices while preserving Dale's law (excitatory/inhibitory signs).
Parameters are specified using matrices for different cell type combinations.

Typical workflow:
    >>> connectivity_graph, cell_indices = sparse_graph_generator(...)
    >>> w_mu_matrix = [[1.0, 1.2], [2.4, 1.6]]  # 2x2 for 2 cell types
    >>> weights = assign_weights_lognormal(connectivity_graph, cell_indices, cell_indices,
    ...                                   [1, -1], w_mu_matrix, w_sigma_matrix)
"""

import numpy as np
from numpy.typing import NDArray

# Type aliases for clarity
IntArray = NDArray[np.int_]
BoolArray = NDArray[np.bool_]
FloatArray = NDArray[np.float64]


def assign_weights_lognormal(
    connectivity_graph: BoolArray,
    source_cell_indices: IntArray,
    target_cell_indices: IntArray,
    cell_type_signs: list,
    w_mu_matrix: list,
    w_sigma_matrix: list,
    seed: int | None = None,
) -> FloatArray:
    """
    Assign log-normal distributed weights to a boolean adjacency matrix.

    Args:
        connectivity_graph (BoolArray): Boolean adjacency matrix.
        source_cell_indices (IntArray): Cell type indices for source neurons (0, 1, 2, ...).
        target_cell_indices (IntArray): Cell type indices for target neurons (0, 1, 2, ...).
        cell_type_signs (list): Signs for each cell type (+1/-1).
        w_mu_matrix (list): NxN matrix of log-normal means.
        w_sigma_matrix (list): NxN matrix of log-normal stds.
        seed (int | None): Random seed for reproducibility.

    Returns:
        FloatArray: Weighted adjacency matrix with log-normal magnitudes and appropriate signs.
    """
    rng = np.random.default_rng(seed)
    w_mu_np = np.array(w_mu_matrix)
    w_sigma_np = np.array(w_sigma_matrix)
    
    W = np.zeros_like(connectivity_graph, dtype=np.float64)
    n_source, n_target = connectivity_graph.shape
    
    # Assign weights based on cell type connectivity matrix
    for i in range(n_source):
        for j in range(n_target):
            if connectivity_graph[i, j]:
                source_type = source_cell_indices[i]
                target_type = target_cell_indices[j]
                
                # Get weight parameters for this cell type combination
                mu = w_mu_np[source_type, target_type]
                sigma = w_sigma_np[source_type, target_type]
                
                # Generate weight magnitude and apply sign (Dale's law)
                weight_magnitude = rng.lognormal(mu, sigma)
                source_sign = cell_type_signs[source_type]
                W[i, j] = weight_magnitude * source_sign
    
    return W


def assign_weights_gamma(
    connectivity_graph: BoolArray,
    source_cell_indices: IntArray,
    target_cell_indices: IntArray,
    cell_type_signs: list,
    shape_matrix: list,
    scale_matrix: list,
    seed: int | None = None,
) -> FloatArray:
    """
    Assign gamma-distributed weights to a boolean adjacency matrix.

    Args:
        connectivity_graph (BoolArray): Boolean adjacency matrix.
        source_cell_indices (IntArray): Cell type indices for source neurons (0, 1, 2, ...).
        target_cell_indices (IntArray): Cell type indices for target neurons (0, 1, 2, ...).
        cell_type_signs (list): Signs for each cell type (+1/-1).
        shape_matrix (list): NxN matrix of gamma shape parameters (k).
        scale_matrix (list): NxN matrix of gamma scale parameters (θ).
        seed (int | None): Random seed for reproducibility.

    Returns:
        FloatArray: Weighted adjacency matrix with gamma-distributed magnitudes and appropriate signs.
    """
    rng = np.random.default_rng(seed)
    shape_np = np.array(shape_matrix)
    scale_np = np.array(scale_matrix)
    
    W = np.zeros_like(connectivity_graph, dtype=np.float64)
    n_source, n_target = connectivity_graph.shape
    
    # Assign weights based on cell type connectivity matrix
    for i in range(n_source):
        for j in range(n_target):
            if connectivity_graph[i, j]:
                source_type = source_cell_indices[i]
                target_type = target_cell_indices[j]
                
                # Get weight parameters for this cell type combination
                shape = shape_np[source_type, target_type]
                scale = scale_np[source_type, target_type]
                
                # Generate weight magnitude and apply sign (Dale's law)
                weight_magnitude = rng.gamma(shape, scale)
                source_sign = cell_type_signs[source_type]
                W[i, j] = weight_magnitude * source_sign
    
    return W


def assign_weights_uniform(
    connectivity_graph: BoolArray,
    source_cell_indices: IntArray,
    target_cell_indices: IntArray,
    cell_type_signs: list,
    low_matrix: list,
    high_matrix: list,
    seed: int | None = None,
) -> FloatArray:
    """
    Assign uniformly distributed weights to a boolean adjacency matrix.

    Args:
        connectivity_graph (BoolArray): Boolean adjacency matrix.
        source_cell_indices (IntArray): Cell type indices for source neurons (0, 1, 2, ...).
        target_cell_indices (IntArray): Cell type indices for target neurons (0, 1, 2, ...).
        cell_type_signs (list): Signs for each cell type (+1/-1).
        low_matrix (list): NxN matrix of uniform distribution lower bounds.
        high_matrix (list): NxN matrix of uniform distribution upper bounds.
        seed (int | None): Random seed for reproducibility.

    Returns:
        FloatArray: Weighted adjacency matrix with uniformly distributed magnitudes and appropriate signs.
    """
    rng = np.random.default_rng(seed)
    low_np = np.array(low_matrix)
    high_np = np.array(high_matrix)
    
    W = np.zeros_like(connectivity_graph, dtype=np.float64)
    n_source, n_target = connectivity_graph.shape
    
    # Assign weights based on cell type connectivity matrix
    for i in range(n_source):
        for j in range(n_target):
            if connectivity_graph[i, j]:
                source_type = source_cell_indices[i]
                target_type = target_cell_indices[j]
                
                # Get weight parameters for this cell type combination
                low = low_np[source_type, target_type]
                high = high_np[source_type, target_type]
                
                # Generate weight magnitude and apply sign (Dale's law)
                weight_magnitude = rng.uniform(low, high)
                source_sign = cell_type_signs[source_type]
                W[i, j] = weight_magnitude * source_sign
    
    return W
